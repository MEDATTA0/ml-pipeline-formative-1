import os

mongo_url = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
mongo_db_name = "stocks_db"
mongo_collection_name = "stocks"
