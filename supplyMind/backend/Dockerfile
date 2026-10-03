FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps (without heavy AI packages — those come from the mounted /ai_src volume)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] sqlalchemy alembic psycopg2-binary \
    pydantic pydantic-settings python-multipart python-jose[cryptography] \
    passlib[bcrypt] redis websockets httpx pandas numpy openpyxl \
    aiofiles python-dotenv reportlab aiosqlite

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
