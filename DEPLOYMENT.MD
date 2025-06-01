# Deployment Architecture

This document outlines the deployment architecture for the Trading Platform application. It is based on the plan detailed in `PLAN_ROBUST_ARCHITECTURE.md`.

## Core Components

The platform is designed as a distributed system with the following core components:

1.  **Web Server (FastAPI with Uvicorn/Gunicorn):**
    *   **Role:** Handles incoming HTTP API requests from users and the frontend application.
    *   **Technology:** Built with FastAPI, a modern, fast (high-performance) web framework for building APIs with Python.
    *   **Deployment:** Typically run using ASGI servers like Uvicorn (for development and some production scenarios) or Gunicorn with Uvicorn workers (for more robust production deployments). Multiple instances can be run behind a load balancer for scalability and high availability.

2.  **Task Queue Broker (Redis):**
    *   **Role:** Manages the communication and distribution of background jobs (tasks) between the Web Server and Task Queue Workers. It ensures that tasks are queued reliably and delivered to available workers.
    *   **Technology:** Redis is the primary configured message broker for Celery in this project (see `backend/celery_app.py`). While RabbitMQ is an alternative, Redis is chosen for its speed and simplicity for many Celery use cases.
    *   **Deployment:** Runs as a separate service. For production, a managed Redis service or a resilient self-hosted setup is recommended.

3.  **Task Queue Workers (Celery):**
    *   **Role:** These are separate Python processes that consume tasks from the Task Queue Broker. They execute long-running or computationally intensive jobs, such as:
        *   Live trading strategy execution loops.
        *   Backtesting strategy simulations.
        *   Potentially other background tasks like sending emails or processing large data.
    *   **Technology:** Celery is used for managing these distributed tasks.
    *   **Deployment:** Workers run independently of the Web Server. Multiple worker processes can be run, even across multiple machines, to scale task processing capacity.

4.  **Database (PostgreSQL):**
    *   **Role:** The central data store for the application. It holds all persistent data, including:
        *   User accounts and profiles.
        *   API key configurations (encrypted).
        *   Strategy definitions and user subscriptions.
        *   Live trade orders and positions.
        *   Backtest results.
        *   Payment and referral information.
    *   **Technology:** PostgreSQL is the specified database (see `backend/.env.example` and database setup in `backend/db.py` which uses SQLAlchemy, compatible with PostgreSQL).
    *   **Deployment:** Runs as a separate service. For production, a managed PostgreSQL service (e.g., AWS RDS, Google Cloud SQL, Azure Database for PostgreSQL) or a robust, replicated self-hosted setup is recommended for data integrity and availability.

## Component Interaction Diagram

The following diagram illustrates how these components interact:

```mermaid
graph TD
    UserBrowser -- HTTP API Requests --> WebServer[Web Server (FastAPI + Uvicorn/Gunicorn)]
    WebServer -- Sends Jobs (e.g., Start Strategy, Run Backtest) --> TaskQueueBroker[Task Queue Broker (Redis)]
    TaskQueueBroker -- Distributes Jobs --> TaskQueueWorkers[Task Queue Workers (Celery)]
    
    TaskQueueWorkers -- Execute --> LiveStrategies[Live Trading Strategies]
    TaskQueueWorkers -- Execute --> Backtests[Backtesting Jobs]
    
    LiveStrategies -- Records/Reads Trade Data --> Database[(PostgreSQL Database)]
    Backtests -- Stores Results/Reads Data --> Database
    LiveStrategies -- Places Orders/Gets Market Data --> Exchanges((External Exchanges via CCXT))
    Backtests -- Gets Historical Market Data --> Exchanges

    WebServer -- Reads/Writes User Data, Configs, etc. --> Database
    
    SecretsManagement[Secrets Management (.env / Vault / Cloud KMS)] -- Provides Credentials --> WebServer
    SecretsManagement -- Provides Credentials --> TaskQueueWorkers
    
    Monitoring[Monitoring & Alerting System] -- Collects Data From --> WebServer
    Monitoring -- Collects Data From --> TaskQueueBroker
    Monitoring -- Collects Data From --> TaskQueueWorkers
    Monitoring -- Collects Data From --> Database
```

## Scalability Considerations

*   **Web Server:** Multiple instances of the FastAPI application can be run behind a load balancer to distribute incoming API traffic.
*   **Task Queue Workers:** The number of Celery worker processes (and potentially the number of machines running workers) can be scaled up or down based on the task load (e.g., number of active strategies, backtests).
*   **Database:** Database scalability can be achieved through read replicas, connection pooling, and choosing appropriate instance sizes if using a managed service.
*   **Redis Broker:** Redis can be configured in a cluster or sentinel setup for high availability and scalability if needed, though for many Celery workloads, a single robust instance suffices.

## Configuration and Secrets

All sensitive configuration, including database URLs, API keys, and secret keys, must be managed via environment variables. Refer to `backend/SECRETS.md` for detailed guidance on setting up environment variables for local development and recommendations for secure secrets management in production environments.

## Running with Docker Compose (Local Development)

This project includes Dockerfiles and a `docker-compose.yml` configuration to simplify local development and testing by orchestrating all backend components.

**Prerequisites:**

*   Docker installed ([https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/))
*   Docker Compose installed (usually comes with Docker Desktop, or can be installed separately: [https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/))

**Setup:**

1.  **Environment File:**
    *   In the `backend/` directory, copy the example environment file:
        ```bash
        cp backend/.env.example backend/.env
        ```
    *   Edit `backend/.env` and fill in your local configuration details. For Docker Compose, ensure:
        *   `DATABASE_URL` points to the `db` service: `postgresql://user:password@db:5432/trading_platform` (replace `user`, `password`, and `trading_platform` with the values you set for `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` in `docker-compose.yml` or your `.env` file).
        *   `REDIS_URL` points to the `redis` service: `redis://redis:6379/0`.
        *   Generate and set strong unique values for `JWT_SECRET_KEY` and `API_ENCRYPTION_KEY`. Refer to `backend/SECRETS.md` for generation commands.

2.  **Build and Run Containers:**
    *   Navigate to the project root directory (where `docker-compose.yml` is located).
    *   Run the following command to build the images (if they don't exist or have changed) and start all services:
        ```bash
        docker-compose up --build
        ```
    *   To run in detached mode (in the background), add the `-d` flag:
        ```bash
        docker-compose up --build -d
        ```

3.  **Database Migrations:**
    *   The `web` service in `docker-compose.yml` is configured to automatically run `alembic upgrade head` upon startup. This ensures your database schema is up-to-date before the web application starts.
    *   If you need to run migrations manually (e.g., to create a new revision or check status), you can execute commands within the running `web` container:
        ```bash
        docker-compose exec web alembic current # Check current revision
        docker-compose exec web alembic revision -m "your_migration_message" # Create new revision
        docker-compose exec web alembic upgrade head # Apply migrations (if not done automatically)
        ```

4.  **Accessing Services:**
    *   **Web Application (API):** `http://localhost:8000`
    *   **PostgreSQL Database:** `localhost:5432` (if port is mapped in `docker-compose.yml`)
    *   **Redis:** `localhost:6379` (if port is mapped)

5.  **Stopping Services:**
    *   If running in the foreground, press `Ctrl+C`.
    *   If running in detached mode, use:
        ```bash
        docker-compose down
        ```
    *   To stop and remove volumes (deletes database and redis data):
        ```bash
        docker-compose down -v
        ```

**Notes for Development:**
*   The `web` and `worker` services in `docker-compose.yml` mount the `backend/` directory as a volume (`./backend:/app`). This allows for live code reloading during development without needing to rebuild the Docker image for every code change in the backend. You might need to restart the Uvicorn/Celery processes within the containers if significant changes (like installing new packages) are made.

## Deployment Steps Overview (Conceptual - For Production)

1.  **Prepare Infrastructure:** Set up PostgreSQL database and Redis instance (preferably managed services in the cloud).
2.  **Configure Environment:** Ensure all required environment variables (as listed in `backend/SECRETS.md`) are securely set in your production environment for both the Web Server and Celery Worker deployments. Do **not** use `.env` files directly in production images.
3.  **Build Production Docker Images:** Build your web and worker images using their respective Dockerfiles. These images should contain all necessary code and dependencies.
    ```bash
    docker build -f backend/Dockerfile.web -t your_registry/trading_platform_web:latest ./backend
    docker build -f backend/Dockerfile.worker -t your_registry/trading_platform_worker:latest ./backend
    # Push images to a container registry
    docker push your_registry/trading_platform_web:latest
    docker push your_registry/trading_platform_worker:latest
    ```
4.  **Database Migrations (Production):** Run Alembic migrations against your production database. This should be a controlled step in your deployment pipeline.
    ```bash
    # Example: Running migrations from a temporary container or a dedicated migration job
    # docker run --rm -e DATABASE_URL="<your_production_db_url>" your_registry/trading_platform_web:latest alembic upgrade head
    ```
5.  **Deploy Web Server & Workers:** Deploy your containerized application using your chosen orchestration platform (e.g., Kubernetes, Docker Swarm, AWS ECS, Google Cloud Run/GKE, Azure Kubernetes Service). Configure the number of replicas for web and worker services based on expected load.
6.  **Deploy Frontend:** Serve the static files from the `frontend/` directory using a web server (e.g., Nginx) or a static site hosting service. Configure it to proxy API requests to the backend Web Server.
7.  **Set Up Monitoring & Logging:** Implement solutions for centralized logging and monitoring of all components.

This document provides a high-level overview. Detailed deployment steps will vary based on the chosen hosting environment and infrastructure.
