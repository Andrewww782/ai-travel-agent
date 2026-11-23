FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY travel_agent travel_agent
# Mount a volume at /data to keep saved trips across deploys.
ENV DB_PATH=/data/travel.db COOKIE_SECURE=1
RUN mkdir /data && useradd --create-home app && chown app /data
USER app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn travel_agent.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
