
## 📦 Langfuse Docker 部署笔记

### 一、数据挂载情况详解

Langfuse 依赖多个有状态组件，数据持久化通过 Docker 命名卷（Named Volume）实现。你当前的配置中，**PostgreSQL、ClickHouse 和 MinIO 已正确挂载**，但 **Redis 尚未挂载**，存在数据丢失风险。

| 服务 | 容器内路径 | 数据内容与功能 | 当前状态 |
| :--- | :--- | :--- | :--- |
| **postgres** | `/var/lib/postgresql/data` | Langfuse 主数据库：用户账户、项目配置、API 密钥、Prompt 模板、数据集元数据、评分配置等核心业务数据。丢失后需重新配置所有项目。 | ✅ 已挂载 `langfuse_postgres_data` |
| **clickhouse** | `/var/lib/clickhouse` | 可观测性数据：**Traces、Observations、Scores** 等。这是数据量最大的部分，也是 Langfuse 分析能力的核心。 | ✅ 已挂载 `langfuse_clickhouse_data` |
| **clickhouse** | `/var/log/clickhouse-server` | ClickHouse 服务端运行日志，用于排查数据库问题。 | ✅ 已挂载 `langfuse_clickhouse_logs` |
| **minio** | `/data` | S3 兼容对象存储：存放**原始事件 JSON**、**多模态媒体文件**（图片/音频）、**批量导出文件**等。Langfuse 的 `LANGFUSE_S3_EVENT_UPLOAD_*` 和 `LANGFUSE_S3_MEDIA_UPLOAD_*` 均指向此处。 | ✅ 已挂载 `langfuse_minio_data` |
| **redis** | `/data` | **事件队列与缓存**。Web 端接收的事件先入队 Redis，再由 Worker 异步消费写入 ClickHouse。**未挂载时，容器重启会导致队列中未处理的事件丢失**，造成部分 Trace 数据缺失。 | ❌ **未挂载，建议添加** |
| **langfuse-web** | `/app/logs` | Web 应用运行日志，用于排查请求处理问题。容器本身是无状态的，核心数据都在外部存储。 | ❌ 未挂载（可选） |
| **langfuse-worker** | `/app/logs` | Worker 处理日志，用于调试事件消费和写入流程。 | ❌ 未挂载（可选） |

#### 建议修改：为 Redis 添加持久化卷

在 `redis` 服务下添加 `volumes` 配置，并在文件底部 `volumes:` 中声明新卷：

```yaml
  redis:
    # ... 已有配置 ...
    volumes:
      - langfuse_redis_data:/data
    command: >
      --requirepass ${REDIS_AUTH:-myredissecret}
      --appendonly yes   # 启用 AOF 持久化，进一步降低数据丢失风险
```

在文件底部的 `volumes:` 部分追加：

```yaml
  langfuse_redis_data:
    driver: local
```

> **可选**：若希望保留 Web/Worker 日志，也可为 `/app/logs` 添加挂载，但这不影响业务数据。

**备份建议**：所有命名卷的数据实际存储在 Docker 管理的本地目录中（默认 `/var/lib/docker/volumes/`）。如需更直观地管理，可将命名卷改为**绑定挂载**（如 `./data/postgres:/var/lib/postgresql/data`），所有数据集中到 `./data/` 目录下，直接打包备份即可。


### 二、修改界面上传数据的大小限制

Langfuse 中存在**多层大小限制**，需要区分对待。你提到的“界面上传数据”通常指**通过 UI 上传数据集 CSV 文件**，其默认限制为 **10MB**。

#### 1. 界面 CSV 上传限制（10MB）

Langfuse 前端代码中，数据集 CSV 上传的 `MAX_FILE_SIZE_BYTES` 定义为 `1024 * 1024 * 1 * 10`，即 **10MB**。若上传超过 10MB 的 CSV，界面会提示 “File too large, Maximum file size is 10MB”。

**修改方法**：此限制**硬编码在前端代码中**，无法通过环境变量直接调整。如需修改，必须**编辑前端源码**并重新构建镜像：
- 定位文件：`web/src/features/datasets/components/UploadDatasetCsv.tsx`
- 修改 `MAX_FILE_SIZE_BYTES` 的值，然后重新构建 Langfuse Web 镜像。

#### 2. API 请求体大小限制（4.5MB，硬编码）

Langfuse 的**摄取 API（Ingestion API）** 存在一个**硬编码的 4.5MB 请求体上限**。即使通过环境变量放宽了事件/批次大小，**超过 4.5MB 的单个请求仍会被拒绝**（返回 413 错误）。

**修改方法**：此限制硬编码在 API 路由配置中（body parser 的 `sizeLimit`），**无法通过环境变量覆盖**。如需提升，必须**手动修改 API 源码并重新部署**：
- 找到 body parser 配置（如 `sizeLimit: "4.5mb"`），调大数值。
- 此操作风险较高，且版本升级后需重复修改。

#### 3. 事件/批次大小限制（可通过环境变量调整）

Langfuse 提供了环境变量用于控制**事件（event）和批次（batch）** 的大小限制，但**受制于上述 4.5MB 硬编码 API 上限**，实际可调整范围有限。

**修改方法**：在你的 `docker-compose.yml` 中，由于 `langfuse-web` 通过 YAML 锚点 `<<: *langfuse-worker-env` 复用了 `langfuse-worker` 的环境变量，只需在 `langfuse-worker` 的 `environment` 块中添加即可：

```yaml
  langfuse-worker:
    environment: &langfuse-worker-env
      # ... 已有配置 ...
      LANGFUSE_MAX_EVENT_SIZE_BYTES: ${LANGFUSE_MAX_EVENT_SIZE_BYTES:-10485760}   # 10MB
      LANGFUSE_MAX_BATCH_SIZE_BYTES: ${LANGFUSE_MAX_BATCH_SIZE_BYTES:-10485760}   # 10MB
```

修改后需**重启服务**使配置生效：

```bash
docker compose down && docker compose up -d
```

> **注意**：此变量仅对 **4.5MB 以内的请求**有效。若数据本身超过 4.5MB，调整此变量无效，必须修改 API 源码或采用下述替代方案。

#### 4. 推荐方案：使用媒体引用绕过大小限制

对于大文本或多模态文件，**官方推荐的方式是使用“媒体引用”** ：通过 Langfuse SDK 将大文件上传到已配置的对象存储（你的 MinIO），然后在数据集或追踪中仅保留一个轻量的引用链接（`@@@langfuseMedia:...@@@`）。这样可**彻底绕开 API 请求体限制**，是处理大文件的最佳实践。


### 三、总结对照表

| 限制类型 | 默认值 | 可调整方式 | 适用范围 |
| :--- | :--- | :--- | :--- |
| 界面 CSV 上传 | 10MB | 修改前端源码，重新构建镜像 | UI 上传数据集 CSV |
| API 请求体 | 4.5MB（硬编码） | 修改 API 源码，重新部署 | 所有摄取请求 |
| 事件大小 | 可配置 | `LANGFUSE_MAX_EVENT_SIZE_BYTES` | 4.5MB 以内的请求 |
| 批次大小 | 可配置 | `LANGFUSE_MAX_BATCH_SIZE_BYTES` | 4.5MB 以内的请求 |
| 媒体文件上传 | 1GB | `LANGFUSE_S3_MEDIA_MAX_CONTENT_LENGTH` | 通过 SDK 上传到对象存储 |

**核心建议**：对于超过 4.5MB 的数据，不要试图通过环境变量“暴力”调大限制，而应**采用媒体引用方案**，将大文件上传到 MinIO，在 Langfuse 中仅保留引用。这是唯一在自托管和云版本中都稳定可靠的方案。