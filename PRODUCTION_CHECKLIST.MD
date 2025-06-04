# Production Readiness Checklist & Recommendations

This document provides a checklist of crucial considerations and recommendations for deploying and maintaining this trading platform in a production environment. Many of these points expand on the configurations already present in `docker-compose.prod.yml` and `DEPLOYMENT.MD`.

## I. Application Configuration & Deployment
- [ ] **Verify `docker-compose.prod.yml`:** Ensure you are using `docker-compose.prod.yml` for production deployments, as it includes production-specific settings (Gunicorn, no code mounts, etc.).
- [ ] **Environment Variables:** Confirm all required environment variables (see `docker-compose.prod.yml` and `backend/.env.example`) are securely provided to the containers by your deployment platform. This includes:
    - `ENVIRONMENT=production`
    - `DATABASE_URL` (pointing to production DB)
    - `REDIS_URL` (pointing to production Redis)
    - `JWT_SECRET_KEY` (unique, strong key)
    - `API_ENCRYPTION_KEY` (unique, strong key for encrypting sensitive data)
    - `GUNICORN_WORKERS` (e.g., 2 * number_of_cores + 1)
    - `CELERY_CONCURRENCY` (appropriate for your workload, e.g., 8-16 for I/O bound tasks)
    - `CELERY_WORKER_REPLICAS` (in `docker-compose.prod.yml` `deploy` section, e.g., 2-4 or more based on load)
    - SMTP settings for email notifications.
    - Payment gateway keys.
    - `STRATEGIES_DIR` pointing to the correct location.
    - `FRONTEND_URL` and `ALLOWED_ORIGINS` for CORS.
- [ ] **Password Reset Token Expiry:** Confirm `PASSWORD_RESET_TOKEN_EXPIRE_HOURS` (environment variable, defaulting to 1 hour if not set) is set to an appropriate duration for your security policy.
- [ ] **Strategies Directory (`STRATEGIES_DIR`):** Verify this environment variable points to a persistent, secure, and **absolute path** on the server where strategy files are located. This is critical if strategies are managed outside the Docker image or on a Windows host.
- [ ] **Runtime Configuration Checks (Celery):** For critical services like Celery, ensure that essential configurations (e.g., `REDIS_URL` in `celery_app.py`) are not using placeholder values in a production environment. Consider adding runtime validation checks if not already present (similar to those in `config.py` for other settings).
- [ ] **Strategy `payment_options_json` Field:** Understand and correctly configure the `payment_options_json` field for each `Strategy` model instance. This field is critical for defining subscription tiers.
    *   **Format:** It must be a valid JSON string representing a list of payment options.
    *   **Structure:** Each item in the list is an object with the following keys:
        *   `months` (integer): The duration of the subscription period in months.
        *   `price_usd` (float): The price for this period in USD.
        *   `description` (string, optional): A brief description for this payment option (e.g., "1 Month Access", "Annual Plan (Best Value)").
    *   **Example:**
        ```json
        [
          {"months": 1, "price_usd": 19.99, "description": "1 Month Access"},
          {"months": 3, "price_usd": 49.99, "description": "3 Months Access (Save 15%)"},
          {"months": 12, "price_usd": 179.99, "description": "1 Year Access (Best Value)"}
        ]
        ```
    *   **Configuration:** This field is typically set by an administrator via the admin interface when creating or editing a strategy. Ensure the JSON is well-formed and accurately reflects the intended subscription options.
- [ ] **Frontend API URL:** Ensure `frontend/js/config.js` has `window.BACKEND_API_BASE_URL = "";` if serving frontend and backend API via a reverse proxy on the same domain.
- [ ] **Nginx/Reverse Proxy:** Implement Nginx (or similar) as a reverse proxy to:
    - Serve the static frontend files.
    - Proxy API requests to the backend (e.g., `127.0.0.1:8000` as configured in `docker-compose.prod.yml`, or `web:8000` if Nginx is also a Docker container).
    - Implement SSL/TLS for HTTPS (e.g., using Let's Encrypt).
    - Optionally, add security headers, caching rules.
- [ ] **System Settings Configuration:** Critical backend defaults (e.g., for live strategy execution parameters, backtesting default settings like `MAX_BACKTEST_DAYS_SYSTEM`, `DEFAULT_BACKTEST_EXCHANGE_ID`, `DEFAULT_BACKTEST_INITIAL_CAPITAL`, and `REFERRAL_COMMISSION_RATE`) are primarily managed via `SystemSetting` database entries or environment variables. Review these in `config.py` and the admin panel to ensure they are appropriate for production.
- [ ] **Windows Docker Desktop Considerations:** If deploying on Windows using Docker Desktop, ensure Docker Desktop is allocated sufficient resources (CPU, RAM, Disk) via its settings, especially when using the WSL 2 backend. Verify Windows Firewall rules allow access to ports exposed by Docker containers.
- [ ] **Review Frontend Configuration:** Ensure platform-specific configurations (e.g., platform subscription pricing in `frontend/js/subscription.js`, currently hardcoded and marked with `FIXME`) are moved to backend/dynamic configuration where appropriate for production.

## II. Automated Testing
- [ ] **Write Comprehensive Tests:** The project is set up with `pytest` (`backend/tests/`). It is CRITICAL to write comprehensive automated tests:
    - **Unit Tests:** For individual functions, methods, and classes in services, models, and utilities.
    - **Integration Tests:** For interactions between components (e.g., API endpoints and services, services and database). Test API authentication, data validation, and business logic.
    - **Strategy Logic Tests:** Rigorously test each trading strategy's logic, including edge cases and simulated market conditions if possible.
- [ ] **CI/CD Pipeline:** Implement a Continuous Integration/Continuous Deployment (CI/CD) pipeline that automatically runs tests on every commit/PR and automates deployments to staging/production.

## III. Security
- [ ] **Secrets Management:** For sensitive data like `JWT_SECRET_KEY`, `API_ENCRYPTION_KEY`, database credentials, and payment gateway keys, use a dedicated secrets management solution (e.g., HashiCorp Vault, AWS Secrets Manager, Google Cloud Secret Manager, Azure Key Vault) as recommended in `backend/SECRETS.MD`. Avoid relying solely on manually configured environment variables if your platform doesn't offer secure injection and at-rest encryption for them.
- [ ] **Regular Security Audits:** Conduct regular security audits of the codebase and infrastructure.
- [ ] **Dependency Scanning:** Use tools to scan dependencies (`requirements.txt`, frontend libraries) for known vulnerabilities (e.g., `safety`, `npm audit` or GitHub Dependabot).
- [ ] **Input Validation:** Ensure all user inputs (API requests, forms) are strictly validated on the backend.
- [ ] **Rate Limiting:** Implement rate limiting on API endpoints to prevent abuse.
- [ ] **Web Application Firewall (WAF):** Consider using a WAF for an additional layer of security.
- [ ] **Permissions & Access Control:** Ensure database users and application roles have the minimum necessary permissions.
- [ ] **Password Reset Tokens:** Password reset tokens are now hashed in the database using the same strong hashing algorithm as user passwords, enhancing security if the database is compromised.

## IV. Monitoring & Logging
- [ ] **Centralized Logging:** Implement centralized logging for all services (web, worker).
    - *Suggestions:* ELK Stack (Elasticsearch, Logstash, Kibana), Grafana Loki with Promtail.
- [ ] **Application Performance Monitoring (APM):**
    - *Suggestions:* Prometheus with Grafana for metrics, Sentry for error tracking, or commercial APM tools.
- [ ] **Celery Monitoring:** Set up Celery Flower or a similar tool to monitor Celery tasks, queues, and workers.
    - Add `flower` to `requirements.txt` and include it in your `docker-compose.prod.yml` if desired.
- [ ] **Infrastructure Monitoring:** Monitor CPU, memory, disk, and network usage of your servers/containers.
- [ ] **Alerting:** Configure alerts for critical errors, performance degradation, and security events.
- [ ] **Error Handling Refinements:** Backend error handling has been improved with more specific exception catching (e.g., for database, Celery, CCXT, payment gateway interactions) and enhanced logging detail (including tracebacks for unexpected errors), aiding in faster diagnosis and debugging.

## V. Database Management
- [ ] **Production Database:** Use a robust production-grade PostgreSQL instance (e.g., managed cloud service like AWS RDS, Google Cloud SQL, or a self-hosted, well-configured instance).
- [ ] **Performance Tuning:** Monitor and tune database performance (query optimization, indexing).
- [ ] **Regular Backups:** Implement and regularly test a reliable database backup and recovery strategy.
- [ ] **Connection Pooling:** Ensure database connection pooling is appropriately configured for the expected load (FastAPI/SQLAlchemy usually handles this, but settings might need tuning).
- [ ] **Store Raw Payment Gateway Metadata:** The `PaymentTransaction` model now includes a `gateway_metadata_json: Text` field. This field stores the raw JSON metadata received from the payment gateway (e.g., during charge creation or via webhooks). This is crucial for robust debugging, manual recovery processes, and detailed payment analysis, reducing reliance on parsing such data from the `description` field.

## VI. Scalability & Reliability
- [ ] **Review Scalability Guide:** Consult `SCALABILITY_AND_PERFORMANCE.MD` for detailed guidance on resource estimation, Docker scaling strategies, performance tuning, and considerations for handling ~5000 concurrent users.
- [ ] **Load Testing:** Conduct load testing to understand performance bottlenecks and determine appropriate scaling for Gunicorn workers, Celery workers, and database resources, as outlined in the scalability guide.
- [ ] **Horizontal Scaling:** Be prepared to scale web and worker services horizontally (running more containers) based on load, as discussed in `SCALABILITY_AND_PERFORMANCE.MD`.
- [ ] **Database & Redis Scaling:** For high-load scenarios, consider using external/managed PostgreSQL and Redis services instead of running them within Docker on the same VPS to improve scalability, reliability, and manageability. (See `SCALABILITY_AND_PERFORMANCE.MD`)
- [ ] **Database Read Replicas:** For read-heavy workloads, consider using database read replicas.
- [ ] **Data Redundancy & Failover:** Plan for data redundancy and service failover.

## VII. Admin Panel Functionality
- [ ] **System Settings Management:** The admin panel now supports viewing and dynamically updating database-backed system settings (e.g., `REFERRAL_COMMISSION_RATE`, `MAX_BACKTEST_DAYS_SYSTEM`, etc.), allowing for easier configuration changes without code deployments.
- [ ] **Strategy Management:** Admin users can now toggle the `is_active` status of strategies and delete strategies directly via the admin interface (note: deleting strategies with active subscriptions is currently allowed but logged).

This checklist is a starting point. Adapt and expand it based on your specific operational requirements and risk assessment.
