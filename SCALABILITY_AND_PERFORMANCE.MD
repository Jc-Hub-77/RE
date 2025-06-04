# Scalability and Performance Guide

## Introduction

This document discusses system resource requirements, scalability strategies, and performance considerations for deploying the Trading Platform. The primary focus is on a Dockerized deployment, which can be hosted on a powerful single Virtual Private Server (VPS) running Linux (recommended) or Windows Server with Docker Desktop.

The initial target for these estimates and strategies is to support approximately **~5000 concurrent users** who are primarily engaged in live trading, with occasional use of backtesting features.

**Disclaimer:** The resource estimates and strategy discussions provided here are initial guidelines. Real-world performance will vary based on actual user behavior, the complexity and number of concurrently running trading strategies, market data volume, and specific configurations. **Comprehensive monitoring and realistic load testing are crucial** to determine the optimal setup for your production environment.

## 1. Estimated System Resources (Dockerized on a Single Powerful VPS)

These estimates assume all components (backend, database, Redis, Celery workers, Nginx) are running as Docker containers on a single, robust VPS.

*   **CPU:**
    *   **FastAPI/Gunicorn `web` service:** FastAPI is highly efficient. Gunicorn worker processes will scale with CPU cores. CPU usage will depend on request volume and complexity of request processing.
    *   **Celery `worker` service:** Can be CPU-intensive, especially if individual trading strategies perform complex calculations or if many strategies are active concurrently per worker replica.
    *   **`db` (PostgreSQL):** Moderate to high CPU usage, depending on query complexity, indexing efficiency, number of concurrent connections, and background maintenance tasks.
    *   **`redis`:** Generally low CPU footprint.
    *   **`nginx` (if containerized as reverse proxy):** Typically very low CPU usage.
    *   **Recommendation:** **8-16+ vCPUs**. For Windows Docker Desktop, ensure the WSL 2 VM (or Hyper-V VM) is allocated a significant portion of these cores.

*   **RAM:**
    *   **FastAPI/Gunicorn `web` service:** Moderate. Each Gunicorn worker will consume RAM. Total usage depends on the number of workers and the data handled per request.
    *   **Celery `worker` service:** Can be a significant RAM consumer. Each Celery worker process (and its child processes/threads depending on concurrency settings) will load strategy code and market data. Many concurrent strategies or strategies handling large datasets will increase RAM needs.
    *   **`db` (PostgreSQL):** Requires substantial RAM for optimal performance, primarily for caching (shared_buffers, effective_cache_size). A common recommendation is to allocate at least 25% of total system RAM to PostgreSQL if it's a dedicated database server, but adjust based on other services.
    *   **`redis`:** RAM usage depends on the amount of data stored (Celery task queues, task results if stored, other potential caching uses). Redis is memory-efficient but can grow if queues become very long or many results are kept.
    *   **Operating System and Docker Overhead:** Account for RAM used by the host OS and the Docker engine itself.
    *   **Recommendation:** **32GB - 64GB+ RAM**. For Windows Docker Desktop, ensure the WSL 2 VM has a generous memory allocation. Insufficient RAM is a common performance bottleneck.

*   **Disk:**
    *   **Storage Type:** **Fast SSD/NVMe storage is crucial** for database performance (IOPS), Docker image storage, and log writing.
    *   **Capacity Breakdown:**
        *   Operating System, Docker images, application code.
        *   PostgreSQL data (`pgdata` volume): This can grow very large depending on trade history, market data logs, user activity logs, and other application data. **This is often the largest and most critical storage component.**
        *   Redis data (if RDB snapshots or AOF persistence is enabled beyond default).
        *   Application logs (from `web`, `worker`, `nginx` containers).
    *   **Recommendation:** **500GB - 1TB+ SSD/NVMe** as a starting point. Implement robust monitoring for disk space, especially for the PostgreSQL data volume, and have a clear plan for expanding storage or archiving old data if necessary.

*   **Network:**
    *   **Bandwidth:** Sufficient bandwidth to handle API requests from users, market data feeds to strategies, and communication with external exchanges.
    *   **Latency:** A high-speed, low-latency internet connection is critical, especially for live trading strategies to minimize slippage and react quickly to market changes. Proximity or good peering to the exchanges your users will trade on is beneficial.

## 2. Scalability Strategies (Docker Compose on a Single VPS)

While a single VPS has ultimate limits, Docker Compose provides mechanisms to scale services vertically (more resources per container, where applicable) and horizontally (more container instances).

*   **Web Service (`web` - FastAPI/Gunicorn):**
    *   **Gunicorn Workers:** Increase the number of Gunicorn worker processes by setting the `GUNICORN_WORKERS` environment variable for the `web` service. A common starting point is `(2 * CPU_CORES) + 1`.
    *   **Replicas (Horizontal Scaling):** In `docker-compose.prod.yml`, under the `web` service's `deploy` section (if using Docker Swarm mode, or just run multiple instances if not in Swarm and your proxy can balance), you can increase `replicas`. Nginx (acting as a reverse proxy) will then distribute incoming API requests across these replicas.
        ```yaml
        # Example for docker-compose.prod.yml (conceptual for Swarm or similar)
        # services:
        #   web:
        #     deploy:
        #       replicas: 3 # Runs 3 instances of the web service
        ```

*   **Celery Workers (`worker`):**
    *   **Concurrency per Worker:** Adjust the `CELERY_CONCURRENCY` environment variable for each `worker` service replica. This controls the number of child processes/threads each Celery worker container can use. Ideal values depend on whether tasks are CPU-bound or I/O-bound.
    *   **Replicas (Horizontal Scaling):** Increase the number of `worker` service replicas in `docker-compose.prod.yml` (e.g., by setting `CELERY_WORKER_REPLICAS` environment variable which is then used in the `deploy.replicas` section or by directly setting `deploy.replicas`). This will run multiple Celery worker containers, distributing task consumption from the Redis queue.
    *   **Queue Separation (Advanced):** For very high loads or to prioritize certain tasks, consider creating separate Celery queues (e.g., `live_trading_queue`, `backtesting_queue`, `email_queue`). You would then run dedicated worker pools for each queue:
        ```bash
        # Worker for live trading tasks
        celery -A backend.celery_app worker -l info -Q live_trading_queue -c 4
        # Worker for backtesting tasks
        celery -A backend.celery_app worker -l info -Q backtesting_queue -c 2
        ```
        This requires changes to your Celery task routing configuration and `docker-compose.prod.yml` to define these specialized worker services.

*   **Database (`db` - PostgreSQL Container):**
    *   **Internal Tuning (Advanced):** Optimize PostgreSQL configuration parameters within `postgresql.conf` (this file would need to be mounted into the container or configured via command-line arguments to `postgres`). Key parameters include `shared_buffers` (e.g., 25% of container's RAM), `work_mem`, `effective_cache_size`. This requires careful tuning and understanding of PostgreSQL.
    *   **Resource Allocation:** Ensure the Docker host provides sufficient dedicated RAM and CPU resources for the PostgreSQL container.
    *   **Limitation:** Scaling a single PostgreSQL container instance on one VPS has inherent limitations. Write performance is particularly hard to scale beyond a single primary node.

*   **Redis (`redis` Container):**
    *   Redis is generally very efficient. Ensure it has enough memory allocated for its data (queues, results, cache).
    *   If Redis becomes a bottleneck (unlikely for this scale unless queues are extremely large or complex Redis operations are used), options include Redis Cluster (complex) or using a managed Redis service.

*   **Nginx (Reverse Proxy - if containerized):**
    *   Nginx is highly performant and typically not a bottleneck for this scale on a single VPS if configured correctly for proxying and serving static files. Worker processes can be tuned in `nginx.conf` if necessary.

## 3. Beyond Single VPS / Advanced Scalability (Brief Mention)

When the limits of a single VPS are reached, or for higher availability and fault tolerance, consider these strategies:

*   **External Managed Databases/Redis:** Offload PostgreSQL and Redis to managed cloud services (e.g., AWS RDS, Google Cloud SQL, Azure Database for PostgreSQL, ElastiCache for Redis, Memorystore). These services offer easier scaling, backups, and maintenance.
*   **Container Orchestration:** For deploying and managing the application across multiple server nodes (a cluster), tools like **Kubernetes** or **Docker Swarm** become necessary. They handle service discovery, load balancing, auto-scaling, and rolling updates.
*   **Load Balancers:** Use dedicated cloud load balancers (e.g., AWS ELB/ALB, Google Cloud Load Balancing) to distribute traffic across multiple instances of your `web` (FastAPI) and `nginx` services running on different nodes.
*   **Database Read Replicas:** For read-heavy database workloads, introduce PostgreSQL read replicas to distribute read queries, reducing load on the primary write database.

## 4. Windows Docker Desktop Specifics

If deploying on Windows Server using Docker Desktop:

*   **Resource Allocation:**
    *   In Docker Desktop settings, ensure you have allocated sufficient CPU cores, memory, and disk space to the WSL 2 VM (or Hyper-V VM if using the older backend). Docker Desktop's default limits might be too low for a production workload of this scale.
    *   Monitor the resource usage of the `vmmem` (WSL 2) or `vmwp` (Hyper-V) process in Windows Task Manager.

*   **WSL 2 Performance:**
    *   WSL 2 generally offers good performance. However, file system I/O performance for files mounted from the Windows host file system into Linux containers (volume mounts like `./frontend:/usr/share/nginx/html`) can sometimes be slower than using Docker named volumes or files directly within the WSL 2 filesystem. For most parts of this application (e.g., database files stored in Docker volumes), this is less of an issue.

*   **Networking:**
    *   Ensure Docker Desktop's networking configuration (e.g., subnet ranges) does not conflict with other network services on your Windows Server.
    *   Port mapping (`ports:` in `docker-compose.yml`) from the container to the Windows host should work seamlessly.

## 5. Key Bottlenecks and Performance Tuning Areas

Anticipate and monitor these common areas:

*   **Database Performance:** This is often the first bottleneck. Focus on:
    *   Efficient SQL queries (use `EXPLAIN ANALYZE`).
    *   Proper indexing for frequently queried columns and join conditions.
    *   Adequate connection pooling (e.g., configured in SQLAlchemy or via PgBouncer).
    *   Regular database maintenance (VACUUM, ANALYZE).
*   **Celery Task Throughput:**
    *   Optimize Celery worker configuration (concurrency, number of replicas).
    *   Ensure tasks are idempotent and efficient.
    *   Monitor queue lengths in Redis.
*   **Strategy Code Efficiency:**
    *   Individual trading algorithms running via Celery can consume significant resources. Profile and optimize them, especially for memory and CPU usage.
    *   Avoid blocking operations within strategy code if possible.
*   **External API Latency:**
    *   Performance will be impacted by the latency and rate limits of the cryptocurrency exchange APIs your strategies connect to. Implement robust error handling, retries, and respect for exchange rate limits.
*   **Network I/O:**
    *   Ensure sufficient server bandwidth and low latency, particularly for market data ingestion and order execution.

## 6. Monitoring and Load Testing (Crucial)

*   **Reiteration:** The resource estimates and scaling strategies are starting points. **They are not a substitute for thorough testing and monitoring.**
*   **Comprehensive Monitoring:** Implement a robust monitoring solution to track:
    *   **System Resources:** CPU, RAM, disk I/O, disk space, network traffic on the host and within containers. Tools like Prometheus with `node_exporter` and `cadvisor` are excellent.
    *   **Application Metrics:** Request rates, error rates, response times for the FastAPI application. (FastAPI middleware for Prometheus).
    *   **Database Performance:** Query latency, connection counts, cache hit rates, replication lag (if using replicas). (e.g., `pg_stat_statements`, Prometheus exporters for PostgreSQL).
    *   **Celery Queues & Workers:** Queue lengths, task processing times, success/failure rates. (Flower is a good tool for Celery monitoring).
    *   **Log Aggregation:** Centralize logs from all services (e.g., using ELK stack, Grafana Loki, or cloud-based logging services).
*   **Load Testing:**
    *   Before going live with a significant user base, conduct realistic load tests that simulate expected user behavior (API usage, strategy execution, backtesting requests).
    *   Use tools like Locust, k6, or JMeter.
    *   Identify bottlenecks under load and adjust resource allocations or application code accordingly.
    *   Determine the actual user capacity your current setup can handle.
