FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

ENV MODEL=dolly-v2-3b-lora-emphatic-dd
ENV TG_BOT_TOKEN=your_bot_token_here

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy models folder and telegram_bot.py
COPY models models
COPY telegram_bot.py .

# Set entrypoint
ENTRYPOINT ["python", "telegram_bot.py"]