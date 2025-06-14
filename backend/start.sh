#!/bin/bash

echo "Running database migrations..."
# Initialize the database engine using the application's init_db function
python -c "import os; from backend.models import init_db; init_db(os.environ.get('DATABASE_URL'))"

echo "Debugging Alembic path..."
echo "Contents of /app:"
ls /app
echo "Contents of /app/backend:"
ls /app/backend
echo "ALEMBIC_CONFIG: $ALEMBIC_CONFIG"
echo "ALEMBIC_SCRIPT_LOCATION: $ALEMBIC_SCRIPT_LOCATION"

ALEMBIC_CONFIG=/app/backend/alembic.ini ALEMBIC_SCRIPT_LOCATION=/app/backend/alembic /usr/local/bin/alembic upgrade head
if [ $? -ne 0 ]; then
    echo "Database migrations failed!"
    exit 1
fi
echo "Database migrations completed."

echo "Starting Gunicorn server..."
cd /app
gunicorn -k uvicorn.workers.UvicornWorker -c /app/backend/gunicorn_conf.py backend.main:app
