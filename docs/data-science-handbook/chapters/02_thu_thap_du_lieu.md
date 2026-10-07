# CHƯƠNG 2. THU THẬP DỮ LIỆU (DATA COLLECTION & INGESTION)

## 2.1. Bản đồ nguồn dữ liệu

| Nguồn | Ví dụ | Công cụ Python | Lưu ý production |
|---|---|---|---|
| File tĩnh | CSV, Excel, JSON, Parquet | `pandas`, `polars`, `pyarrow` | Ưu tiên **Parquet** (có schema, nén, đọc cột) |
| Cơ sở dữ liệu quan hệ | PostgreSQL, MySQL, SQL Server | `SQLAlchemy`, `psycopg`, `connectorx` | Đọc theo chunk, đẩy tính toán xuống SQL |
| Data Warehouse | BigQuery, Snowflake, Redshift | `google-cloud-bigquery`, `snowflake-connector` | Chi phí theo lượng quét — chọn cột, lọc partition |
| Data Lake | S3, GCS, ADLS | `s3fs`, `gcsfs`, `pyarrow.dataset`, `duckdb` | Partition theo ngày, định dạng Parquet/Delta/Iceberg |
| REST/GraphQL API | CRM, thanh toán | `requests`, `httpx` | Retry, rate-limit, phân trang, timeout |
| Web scraping | Trang công khai | `httpx`, `BeautifulSoup`, `Playwright`, `Scrapy` | Tuân thủ robots.txt & điều khoản sử dụng |
| Streaming | Clickstream, IoT | `kafka-python`, `confluent-kafka` | Idempotency, xử lý dữ liệu đến trễ |
| Dữ liệu phi cấu trúc | Văn bản, ảnh, PDF | `pypdf`, `docling`, `Pillow`, `OpenCV` | Lưu metadata + đường dẫn, không nhét blob vào DB |
| Dữ liệu nhãn thủ công | Annotation | Label Studio, Argilla | Đo độ đồng thuận (Cohen's κ) giữa người gán nhãn |

## 2.2. Đọc file — cơ bản đến tối ưu

```python
import pandas as pd
import polars as pl

# CSV: luôn chỉ định dtype + parse_dates để tránh suy luận sai và tốn RAM
dtypes = {"customer_id": "string", "contract": "category", "region": "category"}
df = pd.read_csv("data/raw/churn.csv", dtype=dtypes, parse_dates=["signup_date"],
                 na_values=["", "NA", "null", "-"], encoding="utf-8")

# File lớn hơn RAM: đọc theo chunk
chunks = pd.read_csv("big.csv", chunksize=500_000, dtype=dtypes)
agg = pd.concat(c.groupby("region", observed=True)["monthly_charges"].sum() for c in chunks)
agg = agg.groupby(level=0).sum()

# Parquet: đọc chỉ các cột cần thiết + lọc (predicate pushdown)
df = pd.read_parquet("data/raw/churn.parquet", columns=["customer_id", "tenure_months", "churn"],
                     filters=[("tenure_months", ">", 6)])

# Polars lazy: tối ưu truy vấn tự động, đa luồng, nhanh hơn pandas 5–20 lần trên dữ liệu lớn
result = (
    pl.scan_parquet("data/raw/*.parquet")
    .filter(pl.col("tenure_months") > 6)
    .group_by("contract")
    .agg(pl.col("churn").mean().alias("churn_rate"), pl.len().alias("n"))
    .collect()
)
```

### Giảm bộ nhớ DataFrame

```python
import numpy as np
import pandas as pd


def optimize_dtypes(df: pd.DataFrame, cat_threshold: float = 0.5) -> pd.DataFrame:
    """Downcast số và chuyển chuỗi ít giá trị sang category. Thường giảm 50–80% RAM."""
    out = df.copy()
    for col in out.select_dtypes(include="integer").columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.select_dtypes(include="float").columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    for col in out.select_dtypes(include="object").columns:
        if out[col].nunique(dropna=True) / max(len(out), 1) < cat_threshold:
            out[col] = out[col].astype("category")
    before, after = df.memory_usage(deep=True).sum(), out.memory_usage(deep=True).sum()
    print(f"Memory: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    return out
```

## 2.3. Đọc từ cơ sở dữ liệu (SQL)

```python
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text

from churn.config import Secrets

engine = create_engine(Secrets().db_url, pool_pre_ping=True, pool_size=5)

QUERY = text("""
    SELECT c.customer_id, c.signup_date, c.contract,
           COUNT(t.ticket_id)              AS support_calls_90d,
           AVG(b.amount)                   AS avg_bill_6m
    FROM customers c
    LEFT JOIN tickets t ON t.customer_id = c.customer_id
         AND t.created_at >= :snapshot - INTERVAL '90 days' AND t.created_at < :snapshot
    LEFT JOIN bills b   ON b.customer_id = c.customer_id
         AND b.bill_date  >= :snapshot - INTERVAL '6 months' AND b.bill_date  < :snapshot
    WHERE c.signup_date < :snapshot
    GROUP BY c.customer_id, c.signup_date, c.contract
""")


def load_features(snapshot: str, chunksize: int = 100_000) -> pd.DataFrame:
    # Tham số hóa (bind params) => chống SQL injection + tái sử dụng execution plan
    with engine.connect() as conn:
        parts = pd.read_sql(QUERY, conn, params={"snapshot": snapshot}, chunksize=chunksize)
        return pd.concat(parts, ignore_index=True)
```

**Nguyên tắc:** đẩy `JOIN/GROUP BY/WHERE` xuống database (gần dữ liệu), chỉ kéo về kết quả đã tổng hợp. Mọi điều kiện thời gian phải `< :snapshot` để đảm bảo point-in-time.

### DuckDB — SQL trên file Parquet/CSV không cần server

```python
import duckdb

con = duckdb.connect()
df = con.execute("""
    SELECT contract, AVG(churn) AS churn_rate, COUNT(*) AS n
    FROM read_parquet('data/raw/*.parquet')
    GROUP BY contract ORDER BY churn_rate DESC
""").df()
```

## 2.4. Thu thập qua API — retry, rate limit, phân trang

```python
import logging
import time
from collections.abc import Iterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)


class APIClient:
    def __init__(self, base_url: str, token: str, rate_per_sec: float = 5.0, timeout: float = 30):
        self.client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        self.min_interval = 1.0 / rate_per_sec
        self._last_call = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def get(self, path: str, params: dict | None = None) -> dict:
        self._throttle()
        resp = self.client.get(path, params=params)
        if resp.status_code == 429:  # Too Many Requests -> tôn trọng Retry-After
            time.sleep(float(resp.headers.get("Retry-After", 5)))
        resp.raise_for_status()
        return resp.json()

    def paginate(self, path: str, page_size: int = 100) -> Iterator[dict]:
        cursor = None
        while True:
            params = {"limit": page_size, **({"cursor": cursor} if cursor else {})}
            payload = self.get(path, params)
            yield from payload["data"]
            cursor = payload.get("next_cursor")
            if not cursor:
                break
```

> Với hàng nghìn request, dùng `httpx.AsyncClient` + `asyncio.Semaphore` để giới hạn số kết nối đồng thời.

## 2.5. Web scraping có trách nhiệm

```python
import urllib.robotparser

import httpx
from bs4 import BeautifulSoup

UA = "ResearchBot/1.0 (contact: data-team@example.com)"


def can_fetch(url: str) -> bool:
    rp = urllib.robotparser.RobotFileParser()
    base = "/".join(url.split("/")[:3])
    rp.set_url(f"{base}/robots.txt")
    rp.read()
    return rp.can_fetch(UA, url)


def scrape_table(url: str) -> list[dict]:
    if not can_fetch(url):
        raise PermissionError(f"robots.txt không cho phép: {url}")
    html = httpx.get(url, headers={"User-Agent": UA}, timeout=20).text
    soup = BeautifulSoup(html, "lxml")
    rows = soup.select("table.data tr")
    headers = [th.get_text(strip=True) for th in rows[0].select("th")]
    return [dict(zip(headers, (td.get_text(strip=True) for td in r.select("td")))) for r in rows[1:]]
```

Trang render bằng JavaScript → dùng `Playwright`. Luôn: giới hạn tốc độ, cache kết quả, không thu thập dữ liệu cá nhân trái phép (tuân thủ **Nghị định 13/2023/NĐ-CP** về bảo vệ dữ liệu cá nhân tại Việt Nam, GDPR nếu có khách hàng EU).

## 2.6. Streaming (Kafka) — tiêu thụ sự kiện

```python
import json

from confluent_kafka import Consumer

consumer = Consumer({
    "bootstrap.servers": "kafka:9092",
    "group.id": "churn-feature-builder",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,          # commit thủ công sau khi xử lý thành công
})
consumer.subscribe(["customer-events"])

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        event = json.loads(msg.value())
        # upsert_feature(event) phải idempotent (xử lý lại không gây sai)
        consumer.commit(msg, asynchronous=False)
finally:
    consumer.close()
```

## 2.7. Data Contract & kiểm tra chất lượng dữ liệu đầu vào

Dữ liệu sai là nguyên nhân số 1 gây sự cố mô hình production. **Validate tại biên (ingestion boundary)**.

```python
# src/churn/data/validate.py
import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column

churn_schema = pa.DataFrameSchema(
    {
        "customer_id": Column(str, unique=True, nullable=False,
                              checks=Check.str_matches(r"^C\d{7}$")),
        "signup_date": Column("datetime64[ns]", Check.le(pd.Timestamp.today())),
        "age": Column(float, Check.in_range(18, 100), nullable=True),
        "tenure_months": Column(int, Check.ge(0)),
        "monthly_charges": Column(float, Check.in_range(0, 2_000)),
        "contract": Column(str, Check.isin(["month-to-month", "one-year", "two-year"])),
        "churn": Column(int, Check.isin([0, 1])),
    },
    checks=[
        # Ràng buộc liên cột
        Check(lambda d: (d["total_charges"] >= d["monthly_charges"] * 0.9).mean() > 0.99,
              error="total_charges không nhất quán với monthly_charges"),
    ],
    strict=False,
    coerce=True,
)


def validate(df):
    # lazy=True: gom TẤT CẢ lỗi thay vì dừng ở lỗi đầu tiên
    return churn_schema.validate(df, lazy=True)
```

> Ghi chú phiên bản: với pandera < 0.20 dùng `import pandera as pa`. Trên dữ liệu thô của handbook, schema này sẽ **bắt được** lỗi `"Month-to-Month"` (sai chuẩn hóa chuỗi) — đúng mục đích: lỗi được phát hiện tại biên, rồi xử lý ở bước làm sạch (Chương 5).

**Các chiều chất lượng dữ liệu cần kiểm tra:**

| Chiều | Câu hỏi | Ví dụ kiểm tra |
|---|---|---|
| Completeness | Thiếu bao nhiêu? | % null mỗi cột < ngưỡng |
| Uniqueness | Có trùng không? | `customer_id` duy nhất |
| Validity | Đúng định dạng/miền giá trị? | tuổi ∈ [18, 100] |
| Consistency | Logic giữa các cột/bảng? | `end_date >= start_date` |
| Timeliness | Dữ liệu có mới? | `max(event_time) > now - 1 day` |
| Volume | Số dòng có bất thường? | ±30% so với trung bình 7 ngày |

Công cụ mạnh hơn cho hệ thống lớn: **Great Expectations**, **Soda Core**, **dbt tests**.

## 2.8. Versioning dữ liệu với DVC

```bash
pip install "dvc[s3]"
dvc init
dvc remote add -d storage s3://my-bucket/dvc-store
dvc add data/raw/churn.parquet          # tạo file churn.parquet.dvc (commit vào git)
git add data/raw/churn.parquet.dvc data/raw/.gitignore
git commit -m "data: snapshot churn 2026-10"
dvc push                                # đẩy dữ liệu thật lên S3
# Quay lại phiên bản cũ:
git checkout <commit> && dvc checkout
```

```yaml
# dvc.yaml — pipeline có cache, chỉ chạy lại bước có thay đổi
stages:
  ingest:
    cmd: python -m churn.data.ingest --config configs/data.yaml
    deps: [src/churn/data/ingest.py, configs/data.yaml]
    outs: [data/raw/churn.parquet]
  featurize:
    cmd: python -m churn.features.build
    deps: [src/churn/features/build.py, data/raw/churn.parquet]
    outs: [data/processed/features.parquet]
  train:
    cmd: python -m churn.models.train --config configs/train.yaml
    deps: [src/churn/models/train.py, data/processed/features.parquet]
    params: [configs/train.yaml:]
    outs: [models/model.joblib]
    metrics: [reports/metrics.json: {cache: false}]
```

## 2.9. Pitfalls khi thu thập dữ liệu

1. **Survivorship bias:** chỉ lấy khách hàng *đang* hoạt động → thiếu hẳn nhóm đã churn.
2. **Dữ liệu bị ghi đè (overwrite):** bảng CRM chỉ lưu trạng thái hiện tại → feature tại quá khứ bị "nhìn trộm tương lai". Cần bảng lịch sử (SCD Type 2) hoặc snapshot.
3. **Lệch múi giờ:** chuẩn hóa mọi timestamp về UTC khi lưu, chuyển `Asia/Ho_Chi_Minh` khi hiển thị.
4. **Encoding tiếng Việt:** luôn `utf-8`; chuẩn hóa Unicode `unicodedata.normalize("NFC", s)` (dữ liệu tiếng Việt hay lẫn NFC/NFD).
5. **Sampling bias:** dữ liệu huấn luyện từ một kênh/khu vực không đại diện cho toàn bộ quần thể.

> **Checklist Chương 2**
> - [ ] Biết rõ nguồn, chủ sở hữu, tần suất cập nhật, độ trễ của mọi bảng dữ liệu.
> - [ ] Truy vấn đảm bảo point-in-time (không dùng dữ liệu sau thời điểm dự đoán).
> - [ ] Có data contract & validation tự động tại bước ingestion.
> - [ ] Dữ liệu thô bất biến, được version (DVC / snapshot có ngày).
> - [ ] Tuân thủ quy định bảo vệ dữ liệu cá nhân; PII được mã hóa/ẩn danh.
