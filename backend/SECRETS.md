# Secure Secrets Management

This document outlines how secrets are managed in this application and provides recommendations for production environments.

## Overview

All sensitive information, such as database credentials, API keys, and secret keys, is loaded from environment variables. This practice avoids hardcoding secrets into the codebase, which is a significant security risk.

For local development, these environment variables can be conveniently managed using a `.env` file located in the `backend/` directory. This file is loaded by the application at startup using the `python-dotenv` library.

**IMPORTANT:** The `.env` file itself should **never** be committed to the version control system (e.g., Git) if it contains real credentials. It is intended for local development convenience only. Ensure `.env` is listed in your `.gitignore` file.

## Environment Variables

The application expects the following environment variables to be set:

### Core Application Settings
-   `DATABASE_URL`: The connection string for your database.
    -   *Example (PostgreSQL)*: `postgresql://user:password@host:port/database`
    -   *Example (SQLite for local dev)*: `sqlite:///./trading_platform_dev.db`
-   `JWT_SECRET_KEY`: A strong, random secret key used for signing JWT tokens. This is critical for API security.
    -   *Generation Example*: `openssl rand -hex 32`
-   `API_ENCRYPTION_KEY`: A Fernet key used to encrypt sensitive data stored in the database, such as exchange API keys.
    -   *Generation Example*: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
-   `FRONTEND_URL`: The base URL of your frontend application. Used for CORS configuration and generating links in emails (e.g., email verification, password reset).
    -   *Example*: `http://localhost:8080` or `https://yourdomain.com`
-   `ALLOWED_ORIGINS`: A comma-separated list of origins allowed to make requests to the backend API (CORS).
    -   *Example*: `http://localhost:8080,http://127.0.0.1:8080,https://yourfrontenddomain.com`
-   `ENVIRONMENT`: (Optional) Specifies the runtime environment.
    -   *Values*: `development`, `production`, `testing`
    -   *Example*: `ENVIRONMENT="development"`

### Celery Settings
-   `REDIS_URL`: The connection URL for your Redis instance, used as the message broker and result backend for Celery.
    -   *Example*: `redis://localhost:6379/0`

### Email (SMTP) Settings
These are required for sending emails (e.g., email verification, password resets, notifications).
-   `SMTP_HOST`: Hostname of your SMTP server.
-   `SMTP_PORT`: Port for the SMTP server (e.g., 587 for TLS, 465 for SSL).
-   `SMTP_USER`: Username for SMTP authentication.
-   `SMTP_PASSWORD`: Password for SMTP authentication.
-   `SMTP_TLS`: Set to `True` or `False` (or `1`/`0`) depending on whether your server uses STARTTLS. Defaults to `True`.
-   `EMAILS_FROM_EMAIL`: The email address application emails will be sent from.
-   `EMAILS_FROM_NAME`: The display name for emails sent by the application.

### Payment Gateway Settings (Example: Coinbase Commerce)
-   `COINBASE_COMMERCE_API_KEY`: Your API key for Coinbase Commerce.
-   `COINBASE_COMMERCE_WEBHOOK_SECRET`: The shared secret for verifying webhooks from Coinbase Commerce.

### Application-Specific URLs (Payment Redirects)
These are typically derived from `FRONTEND_URL` but can be overridden.
-   `APP_PAYMENT_SUCCESS_URL`: URL to redirect users to after a successful payment.
-   `APP_PAYMENT_CANCEL_URL`: URL to redirect users to if they cancel a payment.

### Referral System Settings
-   `REFERRAL_COMMISSION_RATE`: The commission rate for referrals (e.g., `0.10` for 10%).
-   `REFERRAL_MINIMUM_PAYOUT_USD`: The minimum commission amount in USD before a payout can be requested/processed.

## Local Development Setup

1.  **Copy the Example:** Create a file named `.env` in the `backend/` directory by copying `backend/.env.example`.
    ```bash
    cp backend/.env.example backend/.env
    ```
2.  **Edit `.env`:** Open the newly created `backend/.env` file and fill in the placeholder values with your actual local development settings. For sensitive keys like `JWT_SECRET_KEY` and `API_ENCRYPTION_KEY`, generate new, unique values.
3.  **Run the Application:** The application will automatically load variables from `backend/.env` at startup.

## Production Secrets Management Recommendations

Using `.env` files with actual secrets in production is **strongly discouraged** due to security risks (e.g., accidental commits, exposure on the server). More secure methods should be employed:

1.  **Cloud Provider Secrets Managers:**
    *   **Examples:** AWS Secrets Manager, Google Cloud Secret Manager, Azure Key Vault.
    *   **Benefits:** These services provide centralized storage for secrets, encryption at rest, fine-grained access control (IAM integration), audit trails, and often automatic rotation capabilities. Applications fetch secrets from these managers at runtime.

2.  **HashiCorp Vault:**
    *   **Benefits:** A powerful, open-source tool for secrets management. Offers features like dynamic secrets, leasing, revocation, and robust access control. Can be self-hosted or used as a managed service.

3.  **Platform-Provided Environment Variables:**
    *   **Examples:** Docker Swarm secrets, Kubernetes Secrets, environment variables injected by PaaS providers (e.g., Heroku config vars, Vercel environment variables).
    *   **Benefits:** Secrets are injected directly into the application's runtime environment by the hosting platform, avoiding the need for files on disk. Ensure the platform encrypts these at rest and manages access securely.

### Why these are more secure:

*   **Centralized Management:** Secrets are stored in one place, making them easier to manage, rotate, and audit.
*   **Encryption at Rest:** Secrets are encrypted when stored.
*   **Access Control:** Fine-grained permissions can be set to control which applications or users can access specific secrets.
*   **Audit Trails:** Access to secrets is typically logged, providing an audit trail.
*   **Reduced Risk of Exposure:** Secrets are not stored in code repositories or directly on application servers in plain text files.

When deploying to production, choose one of these methods based on your infrastructure and operational capabilities. Ensure that only authorized personnel and services have access to modify or retrieve production secrets.
