# CHƯƠNG 2. THU THẬP DỮ LIỆU (DATA COLLECTION & INGESTION)

**Mục tiêu chương:** lấy dữ liệu từ mọi nguồn phổ biến **một cách hiệu quả, đúng thời điểm (point-in-time), có kiểm soát chất lượng, có phiên bản và hợp pháp**.

## 2.1. Bản đồ nguồn dữ liệu

| Nguồn | Ví dụ | Công cụ Python | Lưu ý production |
|---|---|---|---|
| File tĩnh | CSV, Excel, JSON, Parquet | `pandas`, `polars`, `pyarrow` | Ưu tiên **Parquet**: có schema, nén cột, đọc chọn cột/lọc dòng |
| CSDL quan hệ (OLTP) | PostgreSQL, MySQL, SQL Server | `SQLAlchemy 2.x`, `psycopg`, `connectorx`, `adbc` | Đọc từ **replica**, không truy vấn nặng trên DB chính |
| Data Warehouse (OLAP) | BigQuery, Snowflake, Redshift | client chính thức, `ibis` | Trả tiền theo lượng quét: chọn cột, lọc partition |
| Data Lake / Lakehouse | S3/GCS/ADLS + Parquet/Delta/Iceberg | `pyarrow.dataset`, `duckdb`, `polars`, `deltalake` | Partition theo ngày; bảng có transaction log |
| API (REST/GraphQL) | CRM, cổng thanh toán | `httpx`, `requests`, `tenacity` | Timeout, retry có backoff, rate limit, phân trang, idempotency |
| Web scraping | Trang công khai | `httpx` + `BeautifulSoup`/`lxml`, `Playwright`, `Scrapy` | robots.txt, điều khoản sử dụng, dữ liệu cá nhân |
| Streaming / CDC | Clickstream, thay đổi DB | `confluent-kafka`, Debezium | Ngữ nghĩa giao nhận, thứ tự, dữ liệu đến trễ |
| Phi cấu trúc | Văn bản, ảnh, PDF, âm thanh | `pypdf`, `docling`, `Pillow`, `librosa` | Lưu file trong object storage, metadata trong bảng |
| Nhãn do con người | Annotation | Label Studio, Argilla, CVAT | Hướng dẫn gán nhãn, đo độ đồng thuận |

## 2.2. Định dạng file: lựa chọn có cơ sở

| Định dạng | Kiểu lưu | Schema | Nén | Đọc một phần | Dùng khi |
|---|---|---|---|---|---|
| CSV | Dòng, văn bản | ❌ (phải suy luận) | Kém | ❌ | Trao đổi với con người/hệ thống cũ |
| JSON / JSONL | Dòng, văn bản | ❌ | Kém | ❌ | Dữ liệu lồng nhau, log API |
| Excel | Bảng tính | ❌ | Trung bình | ❌ | Dữ liệu nghiệp vụ nhập tay |
| **Parquet** | **Cột** (row groups → column chunks → pages) | ✅ | Tốt (Snappy/ZSTD) | ✅ cột + lọc theo min/max statistics | **Mặc định cho phân tích** |
| Feather/Arrow IPC | Cột, bộ nhớ | ✅ | Tùy chọn | ✅ | Trao đổi nhanh giữa tiến trình |
| Delta / Iceberg | Parquet + transaction log | ✅ + tiến hóa schema | Tốt | ✅ | Lakehouse: ACID, time-travel |

**Cấu trúc Parquet và vì sao nó nhanh.** Một file Parquet gồm nhiều *row group*; mỗi row group lưu từng cột liên tiếp (*column chunk*) kèm thống kê min/max. Khi đọc với `columns=[...]` thì chỉ các cột cần thiết được đọc (**projection pushdown**). Khi đọc với `filters=[...]`, các row group có min/max nằm ngoài điều kiện bị bỏ qua (**predicate pushdown**).

### Thiết lập dữ liệu mẫu cho chương

```python
from pathlib import Path

import numpy as np
import pandas as pd

from churn.data.synthetic import make_churn_data

Path("data/raw").mkdir(parents=True, exist_ok=True)
df = make_churn_data(n=50_000, seed=42)
df.to_csv("data/raw/churn.csv", index=False)
df.to_parquet("data/raw/churn.parquet", index=False, row_group_size=10_000)   # cần pyarrow

# Lake partition theo năm-tháng đăng ký (hive-style: year_month=2023-01/part-0.parquet)
df.assign(year_month=df["signup_date"].dt.strftime("%Y-%m")).to_parquet(
    "data/lake/churn", partition_cols=["year_month"], index=False)
print(sorted(p.name for p in Path("data/lake/churn").iterdir())[:3], "...")
```

## 2.3. Đọc dữ liệu với pandas: chính xác và tiết kiệm

**[Docs]** pandas User Guide, mục *Scaling to large datasets*, khuyến nghị theo thứ tự: (1) **chỉ đọc dữ liệu cần thiết**: chọn cột, lọc dòng; (2) **dùng kiểu dữ liệu hiệu quả**: `category`, số nguyên/thực nhỏ hơn; (3) **chia nhỏ (chunking)**; (4) khi vẫn không đủ, chuyển sang thư viện khác như Polars, DuckDB, Dask hoặc Spark.

```python
# 1) CSV: khai báo dtype + parse_dates thay vì để pandas tự suy luận (chậm, dễ sai)
dtypes = {"customer_id": "string", "contract": "category", "payment_method": "category",
          "region": "category", "tenure_months": "int16", "support_calls": "int16"}
csv_df = pd.read_csv("data/raw/churn.csv", dtype=dtypes, parse_dates=["signup_date"],
                     usecols=list(dtypes) + ["signup_date", "monthly_charges", "churn"],
                     na_values=["", "NA", "null", "-"], encoding="utf-8")

# 2) Parquet: projection + predicate pushdown
pq_df = pd.read_parquet("data/raw/churn.parquet",
                        columns=["customer_id", "tenure_months", "contract", "churn"],
                        filters=[("tenure_months", ">", 24)])

# 3) Đọc một phần lake theo partition: chỉ các thư mục year_month thỏa điều kiện được mở
lake_df = pd.read_parquet("data/lake/churn", filters=[("year_month", ">=", "2024-01")])

# 4) Chunking: tổng hợp file lớn hơn RAM theo từng phần
agg = None
for chunk in pd.read_csv("data/raw/churn.csv", usecols=["contract", "churn"], chunksize=10_000):
    part = chunk.groupby("contract")["churn"].agg(["sum", "count"])
    agg = part if agg is None else agg.add(part, fill_value=0)
print((agg["sum"] / agg["count"]).round(3))
print(len(csv_df), len(pq_df), len(lake_df))
```

### Kiểu dữ liệu và bộ nhớ

```python
def memory_report(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"dtype": frame.dtypes.astype(str),
                         "MB": frame.memory_usage(deep=True, index=False) / 1e6}).round(3)


def optimize_dtypes(frame: pd.DataFrame, max_cat_ratio: float = 0.05) -> pd.DataFrame:
    """Downcast số; chuỗi có ít giá trị phân biệt -> category. Thường giảm 50–90% RAM."""
    out = frame.copy()
    for c in out.select_dtypes("integer").columns:
        out[c] = pd.to_numeric(out[c], downcast="integer")
    for c in out.select_dtypes("float").columns:
        out[c] = pd.to_numeric(out[c], downcast="float")
    for c in out.select_dtypes(["object", "string"]).columns:
        if out[c].nunique(dropna=True) / max(len(out), 1) < max_cat_ratio:
            out[c] = out[c].astype("category")
    return out


raw = pd.read_csv("data/raw/churn.csv")
opt = optimize_dtypes(raw)
print(f"{raw.memory_usage(deep=True).sum() / 1e6:.1f} MB -> {opt.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(memory_report(opt).sort_values("MB", ascending=False).head())
```

> **Lưu ý về float32.** Downcast `float64 → float32` chỉ còn ~7 chữ số có nghĩa. Không áp dụng cho số tiền cần chính xác tuyệt đối, ID số lớn hay timestamp dạng số.

### pandas 3.0: những thay đổi cần biết

| Thay đổi | Hệ quả | Cách viết đúng |
|---|---|---|
| **Copy-on-Write** luôn bật | Mọi DataFrame/Series lấy ra từ đối tượng khác hành xử như bản sao. **Chained assignment** `df["a"][mask] = v` **không còn sửa `df`** | `df.loc[mask, "a"] = v` |
| Kiểu chuỗi mặc định `str` (dùng PyArrow nếu có) | Cột chuỗi không còn là `object`; thiếu giá trị là `NaN` | Kiểm tra `pd.api.types.is_string_dtype` thay vì `== object` |
| Datetime không còn mặc định nano giây | Độ phân giải được suy luận (thường là `us`) khi đọc/tạo datetime | Không giả định `datetime64[ns]` trong schema/test. Ép kiểu tường minh khi cần |
| Nhiều API cũ bị xóa | Code pandas 1.x có thể lỗi | Chạy test với `-W error::FutureWarning` trên pandas 2.x trước khi nâng cấp |

```python
s = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
print(s.dtypes.to_dict())                        # b là kiểu chuỗi chuyên dụng ở pandas 3
s.loc[s["a"] > 1, "a"] = 0                       # cách duy nhất đúng để gán có điều kiện
print(s["a"].tolist())
```

### `dtype_backend="pyarrow"`: kiểu dữ liệu có hỗ trợ missing thật sự

```python
arrow_df = pd.read_parquet("data/raw/churn.parquet", dtype_backend="pyarrow")
print(arrow_df.dtypes.head(4).to_dict())
# int64[pyarrow] giữ được giá trị thiếu mà không bị ép sang float như numpy int
```

## 2.4. Khi pandas không đủ: Polars và DuckDB

```python
import duckdb
import polars as pl

# Polars lazy: xây kế hoạch truy vấn, tối ưu (pushdown, chạy song song) rồi mới thực thi
lazy = (
    pl.scan_parquet("data/raw/churn.parquet")
    .filter(pl.col("tenure_months") > 6)
    .group_by("contract")
    .agg(pl.col("churn").mean().alias("churn_rate"), pl.len().alias("n"))
    .sort("churn_rate", descending=True)
)
print(lazy.explain().splitlines()[0])            # xem kế hoạch truy vấn đã tối ưu
print(lazy.collect())

# DuckDB: SQL trực tiếp trên Parquet/CSV/DataFrame, không cần server
con = duckdb.connect()
print(con.sql("""
    SELECT contract, round(avg(churn), 4) AS churn_rate, count(*) AS n
    FROM read_parquet('data/raw/churn.parquet')
    WHERE tenure_months > 6
    GROUP BY contract ORDER BY churn_rate DESC
""").df())
```

| Công cụ | Mô hình thực thi | Mạnh nhất khi |
|---|---|---|
| pandas | Eager, phần lớn đơn luồng | Dữ liệu vừa RAM, hệ sinh thái lớn nhất |
| Polars | Lazy + eager, đa luồng, Arrow | Biến đổi dữ liệu lớn trên một máy |
| DuckDB | SQL OLAP nhúng, vectorized | Phân tích SQL trên file, join lớn |
| Spark / Dask | Phân tán | Dữ liệu vượt một máy |

## 2.5. Đọc từ cơ sở dữ liệu với SQLAlchemy 2.x

Nguyên tắc:

1. **Đẩy tính toán xuống database** (JOIN, GROUP BY, WHERE) và chỉ kéo về kết quả.
2. **Tham số hóa truy vấn** bằng bind parameters, không dùng f-string. Cách này vừa chống SQL injection vừa giúp tái sử dụng execution plan.
3. **Point-in-time:** mọi điều kiện thời gian phải `< :snapshot`.
4. Kết nối dùng **pool**, `pool_pre_ping=True` để tự loại kết nối chết. Đọc từ **read replica**.

```python
from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///data/warehouse.db")        # production: postgresql+psycopg://...
customers = df.drop_duplicates("customer_id")[["customer_id", "signup_date", "contract"]]
rng = np.random.default_rng(0)
tickets = pd.DataFrame({
    "ticket_id": np.arange(30_000),
    "customer_id": rng.choice(customers["customer_id"], 30_000),
    "created_at": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 600, 30_000), unit="D"),
})
with engine.begin() as conn:                                 # transaction: commit/rollback tự động
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    tickets.to_sql("tickets", conn, if_exists="replace", index=False)

FEATURE_SQL = text("""
    SELECT c.customer_id,
           c.contract,
           COUNT(t.ticket_id) AS tickets_90d
    FROM customers c
    LEFT JOIN tickets t
           ON t.customer_id = c.customer_id
          AND t.created_at >= :window_start
          AND t.created_at <  :snapshot           -- point-in-time: không nhìn tương lai
    WHERE c.signup_date < :snapshot
    GROUP BY c.customer_id, c.contract
""")


def load_features(snapshot: str, window_days: int = 90) -> pd.DataFrame:
    snap = pd.Timestamp(snapshot)
    params = {"snapshot": str(snap), "window_start": str(snap - pd.Timedelta(days=window_days))}
    with engine.connect() as conn:
        return pd.read_sql(FEATURE_SQL, conn, params=params)


feats = load_features("2024-03-01")
print(feats.shape, feats["tickets_90d"].describe()[["mean", "max"]].round(2).to_dict())
```

### Point-in-time join trong pandas: `merge_asof`

Khi feature được cập nhật theo thời gian (ví dụ hạng thành viên thay đổi), mỗi dòng nhãn tại thời điểm T phải lấy **giá trị gần nhất trước T**, không được lấy giá trị mới nhất.

```python
labels = pd.DataFrame({"customer_id": ["C1", "C1", "C2"],
                       "snapshot_date": pd.to_datetime(["2024-02-01", "2024-05-01", "2024-05-01"])})
tier_history = pd.DataFrame({"customer_id": ["C1", "C1", "C2"],
                             "valid_from": pd.to_datetime(["2024-01-10", "2024-04-15", "2024-06-01"]),
                             "tier": ["silver", "gold", "platinum"]})

pit = pd.merge_asof(
    labels.sort_values("snapshot_date"), tier_history.sort_values("valid_from"),
    left_on="snapshot_date", right_on="valid_from", by="customer_id",
    direction="backward", allow_exact_matches=False,   # chỉ lấy bản ghi có hiệu lực TRƯỚC T
)
print(pit)   # C2 tại 2024-05-01 chưa có tier (NaN): đúng, vì tier platinum có từ 2024-06-01
```

> **[Kinh nghiệm]** Bảng nghiệp vụ thường chỉ lưu **trạng thái hiện tại** (bị ghi đè). Nếu tính feature từ bảng như vậy cho các snapshot quá khứ, bạn đang nhìn trộm tương lai. Cần bảng lịch sử (**SCD Type 2**: `valid_from`, `valid_to`), event log, hoặc snapshot định kỳ. Feature store (Chương 12) giải quyết vấn đề này một cách hệ thống.

## 2.6. Thu thập qua API: timeout, retry, rate limit, phân trang

**[Docs]** HTTPX: luôn đặt **timeout** (mặc định của httpx là 5 giây). Tái sử dụng `Client` để có connection pooling. `tenacity` cung cấp retry có **exponential backoff + jitter**. Chỉ retry các lỗi **tạm thời**: lỗi mạng, 429, 5xx. Không retry lỗi 4xx khác vì đó là lỗi phía client.

```python
import time

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {429, 500, 502, 503, 504}


class APIClient:
    def __init__(self, base_url: str, token: str, rate_per_sec: float = 10.0,
                 transport: httpx.BaseTransport | None = None):
        self.client = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {token}"},
                                   timeout=httpx.Timeout(10.0, connect=3.0), transport=transport)
        self.min_interval, self._last = 1.0 / rate_per_sec, 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    @retry(retry=retry_if_exception(is_transient), wait=wait_random_exponential(multiplier=0.01, max=1),
           stop=stop_after_attempt(5), reraise=True)
    def get(self, path: str, params: dict | None = None) -> dict:
        self._throttle()
        resp = self.client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def paginate(self, path: str, page_size: int = 100):
        cursor = None
        while True:
            payload = self.get(path, {"limit": page_size, **({"cursor": cursor} if cursor else {})})
            yield from payload["data"]
            if not (cursor := payload.get("next_cursor")):
                break


# Kiểm thử không cần server thật: MockTransport giả lập API có phân trang và lỗi 503 thoáng qua
calls = {"n": 0}


def fake_api(request: httpx.Request) -> httpx.Response:
    calls["n"] += 1
    if calls["n"] == 2:
        return httpx.Response(503)                               # lỗi tạm thời -> được retry
    cursor = int(request.url.params.get("cursor", 0))
    data = [{"id": i} for i in range(cursor, min(cursor + 100, 250))]
    nxt = cursor + 100 if cursor + 100 < 250 else None
    return httpx.Response(200, json={"data": data, "next_cursor": nxt})


api = APIClient("https://crm.example.com", token="***", rate_per_sec=1000,
                transport=httpx.MockTransport(fake_api))
records = list(api.paginate("/customers"))
print(len(records), "bản ghi;", calls["n"], "request (có 1 lần retry)")
```

### Thu thập song song có giới hạn (async)

```python
import asyncio


async def fetch_all(ids: list[int], max_concurrency: int = 20) -> list[dict]:
    sem = asyncio.Semaphore(max_concurrency)              # giới hạn số kết nối đồng thời
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"id": r.url.path.split("/")[-1]}))
    async with httpx.AsyncClient(base_url="https://crm.example.com", transport=transport,
                                 timeout=10) as client:
        async def one(i: int) -> dict:
            async with sem:
                r = await client.get(f"/customers/{i}")
                r.raise_for_status()
                return r.json()
        return await asyncio.gather(*(one(i) for i in ids))


print(len(asyncio.run(fetch_all(list(range(500))))))
```

## 2.7. Web scraping có trách nhiệm

```python
from bs4 import BeautifulSoup

HTML = """<table class="plans"><tr><th>Gói</th><th>Giá</th><th>Data</th></tr>
<tr><td>Basic</td><td>70.000</td><td>5 GB</td></tr>
<tr><td>Pro</td><td>150.000</td><td>30 GB</td></tr></table>"""


def parse_plans(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows = soup.select("table.plans tr")
    header = [th.get_text(strip=True) for th in rows[0].select("th")]
    data = [[td.get_text(strip=True) for td in r.select("td")] for r in rows[1:]]
    out = pd.DataFrame(data, columns=header)
    out["Giá"] = out["Giá"].str.replace(".", "", regex=False).astype(int)   # chuẩn hóa định dạng VN
    return out


print(parse_plans(HTML))
```

Quy tắc bắt buộc:

- Kiểm tra `robots.txt` (`urllib.robotparser`) và **điều khoản sử dụng** của trang. Đặt `User-Agent` có thông tin liên hệ.
- Giới hạn tốc độ, cache kết quả, chạy ngoài giờ cao điểm.
- Không thu thập **dữ liệu cá nhân** khi không có cơ sở pháp lý. Tại Việt Nam phải tuân thủ **Nghị định 13/2023/NĐ-CP** về bảo vệ dữ liệu cá nhân; với khách hàng EU là **GDPR**.
- Trang render bằng JavaScript thì dùng **Playwright**. Hệ thống crawl lớn thì dùng **Scrapy** (có sẵn throttling, pipeline, retry).

## 2.8. Streaming & Change Data Capture

| Khái niệm | Ý nghĩa | Hệ quả cho Data Scientist |
|---|---|---|
| At-most-once | Có thể mất sự kiện | Feature đếm bị thiếu |
| **At-least-once** (phổ biến) | Có thể trùng sự kiện | Xử lý phải **idempotent** (dedup theo `event_id`) |
| Exactly-once | Không mất, không trùng | Cần transaction (Kafka EOS, Flink checkpoint) |
| Event time vs processing time | Thời điểm xảy ra vs thời điểm hệ thống nhận | Feature phải dùng **event time** |
| Watermark | Ngưỡng chờ dữ liệu đến trễ | Quyết định khi nào "đóng" cửa sổ tổng hợp |
| CDC (Debezium) | Đọc log thay đổi của DB thành luồng sự kiện | Xây lại lịch sử để có feature point-in-time |

```python norun
import json

from confluent_kafka import Consumer

consumer = Consumer({
    "bootstrap.servers": "kafka:9092",
    "group.id": "churn-feature-builder",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,           # commit thủ công SAU khi xử lý xong -> at-least-once
    "isolation.level": "read_committed",   # bỏ qua message của transaction bị hủy
})
consumer.subscribe(["customer-events"])
try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            raise RuntimeError(msg.error())
        event = json.loads(msg.value())
        upsert_feature(event)              # idempotent: khóa theo event["event_id"]
        consumer.commit(message=msg, asynchronous=False)
finally:
    consumer.close()
```

## 2.9. Data Contract & kiểm tra chất lượng dữ liệu

**Data contract** là thỏa thuận giữa bên sản xuất và bên tiêu thụ dữ liệu về schema, ngữ nghĩa, chất lượng và SLA. Kiểm tra phải đặt **tại biên** (ingestion) và **trước khi chấm điểm**.

### Sáu chiều chất lượng dữ liệu

| Chiều | Câu hỏi | Ví dụ kiểm tra |
|---|---|---|
| Completeness | Thiếu bao nhiêu? | % null mỗi cột < ngưỡng |
| Uniqueness | Có trùng không? | `customer_id` duy nhất |
| Validity | Đúng kiểu, định dạng, miền giá trị? | `age ∈ [18, 100]`, regex mã khách |
| Consistency | Logic giữa cột/bảng? | `total ≈ monthly × tenure`; khóa ngoại tồn tại |
| Timeliness | Dữ liệu có mới? | `max(event_time) > now − 1 ngày` |
| Accuracy | Đúng với thực tế? | Đối soát với nguồn chuẩn (sổ cái kế toán) |

### pandera: schema dạng class (DataFrameModel)

Package tham chiếu định nghĩa contract trong `churn/data/validate.py`:

```python norun
import pandera.pandas as pa
from pandera.typing import Series


class ChurnSchema(pa.DataFrameModel):
    customer_id: Series[str] = pa.Field(unique=True, str_matches=r"^C\d{7}$")
    signup_date: Series[pd.Timestamp] = pa.Field(le=pd.Timestamp("2100-01-01"))
    age: Series[float] = pa.Field(ge=18, le=100, nullable=True)
    tenure_months: Series[int] = pa.Field(ge=0, le=600)
    monthly_charges: Series[float] = pa.Field(gt=0, le=2_000, nullable=True)
    contract: Series[str] = pa.Field(isin=["month-to-month", "one-year", "two-year"])
    churn: Series[int] = pa.Field(isin=[0, 1])
    # ... (xem file đầy đủ)

    @pa.dataframe_check(error="total_charges inconsistent with monthly_charges * tenure")
    def charges_consistent(cls, df: pd.DataFrame) -> bool:
        expected = df["monthly_charges"] * df["tenure_months"]
        ok = (df["total_charges"] - expected).abs() <= 0.05 * expected + 1
        return bool(ok[expected.notna()].mean() > 0.99)

    class Config:
        coerce = True      # ép kiểu trước khi kiểm tra
        strict = False     # cho phép cột thừa
```

Chạy contract trên **dữ liệu thô** (có lỗi cài sẵn) và trên dữ liệu đã làm sạch:

```python
import pandera.errors

from churn.data.validate import validate
from churn.features.clean import clean_churn

try:
    validate(df)                                     # lazy=True: gom TẤT CẢ lỗi một lần
except pandera.errors.SchemaErrors as err:
    fc = err.failure_cases
    print(fc.groupby(["column", "check"], dropna=False).size().sort_values(ascending=False).head(6))

validate(clean_churn(df))                           # sau khi làm sạch: hợp lệ
print("Dữ liệu sạch thỏa mãn contract")
```

Contract phát hiện `customer_id` trùng (100 dòng lặp, tính cả hai bản) và `contract` sai chuẩn hóa (`Month-to-Month`). Lỗi được bắt **tại biên**, rồi xử lý có chủ đích ở bước làm sạch (Chương 5).

> **Bài học quan trọng.** 30 ngoại lai `monthly_charges` ×8 (tối đa ~1 600) **không** bị bắt, vì vẫn nằm trong miền hợp lệ (0; 2 000]. Kiểm tra miền giá trị chỉ bắt giá trị **không thể có**, không bắt được giá trị **bất thường**. Cần thêm kiểm tra thống kê: phân vị, z-score robust, so sánh phân phối với lô tham chiếu (Chương 3.7, 12.9).

### Kiểm tra cấp lô dữ liệu (volume, freshness, phân phối)

```python
def batch_checks(batch: pd.DataFrame, reference_rows: int, max_age_days: int = 2,
                 now: pd.Timestamp | None = None) -> dict[str, bool]:
    now = now or pd.Timestamp.now()
    return {
        "volume_ok": 0.7 * reference_rows <= len(batch) <= 1.3 * reference_rows,
        "fresh_ok": (now - batch["signup_date"].max()).days <= max_age_days,
        "null_rate_ok": bool((batch.isna().mean() < 0.2).all()),
        "target_rate_ok": 0.05 <= batch["churn"].mean() <= 0.4,
    }


print(batch_checks(clean_churn(df), reference_rows=50_000, now=pd.Timestamp("2024-06-20")))
```

**Công cụ cho hệ thống lớn:** Great Expectations (Expectation Suites, Data Docs), Soda Core (SodaCL), dbt tests (`unique`, `not_null`, `relationships`, `accepted_values`), Deequ trên Spark.

## 2.10. Versioning dữ liệu với DVC

**[Docs]** DVC lưu dữ liệu lớn ở remote (S3/GCS/Azure/SSH). Git chỉ giữ file `.dvc` nhỏ chứa hash nội dung (MD5). Nhờ vậy *mỗi commit git xác định chính xác phiên bản dữ liệu*.

```bash
pip install "dvc[s3]"
dvc init
dvc remote add -d storage s3://ml-bucket/dvc-store
dvc add data/raw/churn.parquet                 # sinh data/raw/churn.parquet.dvc
git add data/raw/churn.parquet.dvc data/raw/.gitignore && git commit -m "data: churn snapshot 2026-10"
dvc push                                       # đẩy dữ liệu thật lên remote
git checkout <commit-cũ> && dvc checkout       # quay lại đúng dữ liệu của commit đó
```

```yaml
# dvc.yaml — pipeline có cache: chỉ chạy lại stage có deps/params thay đổi
stages:
  featurize:
    cmd: python -m churn.features.make --in data/raw/churn.parquet --out data/processed/features.parquet
    deps: [src/churn/features, data/raw/churn.parquet]
    outs: [data/processed/features.parquet]
  train:
    cmd: python -m churn.models.train --config configs/train.yaml
    deps: [src/churn/models, data/processed/features.parquet]
    params: [configs/train.yaml:model_params]
    outs: [models/model.joblib]
    metrics: [reports/metrics.json: {cache: false}]
```

`dvc repro` chạy lại pipeline. `dvc metrics diff` và `dvc params diff` so sánh giữa các commit.

## 2.11. Dữ liệu nhãn do con người gán

**[Course]** *MLOps Specialization* (Andrew Ng) nhấn mạnh **tính nhất quán của nhãn** ("label consistency"): nhãn không nhất quán gây hại như nhiễu. Quy trình:

1. Viết **hướng dẫn gán nhãn** có ví dụ biên (edge cases).
2. Cho 2–3 người gán cùng một mẫu, đo **độ đồng thuận**.
3. Thảo luận các ca bất đồng và cập nhật hướng dẫn. Lặp lại tới khi đạt ngưỡng.

```python
from sklearn.metrics import cohen_kappa_score

rng = np.random.default_rng(3)
truth = rng.integers(0, 3, 500)                                  # 3 loại khiếu nại
annot_a = np.where(rng.random(500) < 0.85, truth, rng.integers(0, 3, 500))
annot_b = np.where(rng.random(500) < 0.75, truth, rng.integers(0, 3, 500))
print("Tỷ lệ trùng khớp thô:", (annot_a == annot_b).mean().round(3))
print("Cohen's κ          :", round(cohen_kappa_score(annot_a, annot_b), 3))
# κ hiệu chỉnh cho đồng thuận ngẫu nhiên. Thang Landis & Koch (1977):
# <0.2 kém | 0.21–0.4 khá | 0.41–0.6 trung bình | 0.61–0.8 tốt | >0.8 gần hoàn hảo
```

Với nhiều người gán hoặc nhãn thứ bậc, dùng **Fleiss' κ** hoặc **Krippendorff's α**. Khi nhãn nhiễu, các kỹ thuật *confident learning* (thư viện `cleanlab`) giúp tìm mẫu bị gán sai.

## 2.12. Quyền riêng tư: ẩn danh và giả danh

| Kỹ thuật | Mô tả | Lưu ý |
|---|---|---|
| Giả danh (pseudonymization) | Thay định danh bằng mã, ví dụ HMAC có khóa bí mật | Vẫn là dữ liệu cá nhân theo GDPR/NĐ 13 nếu còn khóa |
| Tổng quát hóa | Tuổi → nhóm tuổi; địa chỉ → tỉnh | Giảm rủi ro tái định danh |
| k-anonymity | Mỗi tổ hợp quasi-identifier xuất hiện ≥ k lần | Không chống được tấn công đồng nhất thuộc tính |
| Differential privacy | Thêm nhiễu có kiểm soát vào thống kê | Chuẩn mực mạnh nhất cho dữ liệu công bố |

```python
import hashlib
import hmac


def pseudonymize(values: pd.Series, secret: bytes) -> pd.Series:
    """HMAC-SHA256: ổn định (cùng input -> cùng mã) để join; không đảo ngược được khi không có khóa."""
    return values.map(lambda v: hmac.new(secret, str(v).encode(), hashlib.sha256).hexdigest()[:16])


def k_anonymity(frame: pd.DataFrame, quasi_identifiers: list[str]) -> int:
    return int(frame.groupby(quasi_identifiers, observed=True).size().min())


SECRET = b"load-from-secret-manager"          # production: lấy từ Vault/Secrets Manager
safe = df[["customer_id", "age", "region"]].dropna().copy()
safe["customer_id"] = pseudonymize(safe["customer_id"], SECRET)
safe["age_band"] = pd.cut(safe["age"], [17, 25, 35, 45, 55, 65, 100]).astype(str)
print("k (age, region)      =", k_anonymity(safe, ["age", "region"]))
print("k (age_band, region) =", k_anonymity(safe, ["age_band", "region"]))
```

## 2.13. Pitfalls khi thu thập dữ liệu

1. **Survivorship bias:** chỉ lấy khách *đang hoạt động* nên thiếu hẳn nhóm đã churn.
2. **Nhìn trộm tương lai qua bảng bị ghi đè:** dùng trạng thái hiện tại để tính feature cho quá khứ.
3. **Múi giờ:** lưu UTC, chỉ đổi sang `Asia/Ho_Chi_Minh` khi hiển thị. Cẩn thận ranh giới ngày khi tổng hợp.
4. **Unicode tiếng Việt:** dữ liệu lẫn NFC/NFD làm `"Hà Nội" != "Hà Nội"`. Chuẩn hóa bằng `unicodedata.normalize("NFC", s)`.
5. **Sampling bias:** dữ liệu từ một kênh/vùng không đại diện cho quần thể chấm điểm.
6. **Thay đổi định nghĩa ở nguồn:** bên sản xuất đổi đơn vị (VNĐ → nghìn VNĐ) mà không báo. Đây là lý do cần data contract và giám sát phân phối.

> **Checklist Chương 2**
> - [ ] Biết nguồn, chủ sở hữu, tần suất cập nhật, độ trễ của mọi bảng.
> - [ ] Dùng định dạng cột (Parquet), chỉ đọc cột/dòng cần; kiểu dữ liệu tối ưu.
> - [ ] Truy vấn tham số hóa; feature đảm bảo point-in-time (`< snapshot`, `merge_asof`).
> - [ ] API client có timeout, retry chỉ cho lỗi tạm thời, rate limit, test bằng mock.
> - [ ] Data contract chạy tự động tại ingestion và trước khi chấm điểm.
> - [ ] Dữ liệu thô bất biến, có phiên bản (DVC/snapshot).
> - [ ] Nhãn thủ công có hướng dẫn và đo độ đồng thuận.
> - [ ] PII được giả danh/ẩn danh; tuân thủ NĐ 13/2023/NĐ-CP.

### Tài liệu tham khảo Chương 2

- pandas User Guide: *IO tools*, *Scaling to large datasets*, *Copy-on-Write*, *PyArrow functionality*. pandas.pydata.org/docs/user_guide
- Apache Parquet. *File format documentation.* parquet.apache.org/docs
- Polars User Guide: *Lazy API.* docs.pola.rs · DuckDB: *Parquet import.* duckdb.org/docs
- SQLAlchemy 2.0. *Unified Tutorial.* docs.sqlalchemy.org/en/20/tutorial
- HTTPX documentation: *Timeouts*, *Transports (MockTransport)*. python-httpx.org · Tenacity docs.
- pandera documentation: *DataFrame Models.* pandera.readthedocs.io
- DVC documentation: *Data versioning*, *Pipelines.* dvc.org/doc
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*, ch. 11 "Stream Processing". O'Reilly.
- Landis, J. R. & Koch, G. G. (1977). *The Measurement of Observer Agreement for Categorical Data.* Biometrics 33(1).
- Chính phủ Việt Nam. *Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân.*
