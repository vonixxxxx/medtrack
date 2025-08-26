# Drug Information API - FastAPI Backend

A FastAPI backend that integrates with the OpenFDA Drugs API to provide drug information by searching for active ingredients.

## Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **OpenFDA Integration**: Searches the FDA's drug database for comprehensive information
- **Async Operations**: Uses async/await for better performance
- **Error Handling**: Comprehensive error handling with proper HTTP status codes
- **CORS Support**: Configured to work with React frontend
- **Type Hints**: Full Python type annotations for better code quality

## API Endpoints

### GET `/`
- **Description**: Root endpoint with welcome message
- **Response**: JSON with API information

### GET `/drug/{drug_name}`
- **Description**: Search for drug information by active ingredient
- **Parameters**: `drug_name` (string) - The drug/active ingredient to search for
- **Response**: JSON with drug information including:
  - `brand_name`: The brand name of the drug
  - `generic_name`: The generic name of the drug
  - `indications_and_usage`: What the drug is used for
  - `warnings`: Important warnings about the drug

### GET `/health`
- **Description**: Health check endpoint
- **Response**: JSON with API status

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Navigate to the backend directory**:
   ```bash
   cd backend_fastapi
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Alternative: Using Docker

If you have Docker installed:

```bash
# Build the Docker image
docker build -t drug-api .

# Run the container
docker run -p 8000:8000 drug-api
```

## Usage Examples

### Search for Aspirin
```bash
curl "http://localhost:8000/drug/aspirin"
```

### Search for Ibuprofen
```bash
curl "http://localhost:8000/drug/ibuprofen"
```

### Health Check
```bash
curl "http://localhost:8000/health"
```

## API Documentation

Once the server is running, you can access:

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Error Handling

The API returns appropriate HTTP status codes:

- **200**: Success - Drug information found
- **404**: Not Found - No drug information found
- **500**: Internal Server Error - API or network error

## Dependencies

- **FastAPI**: Web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **HTTPX**: Async HTTP client for external API calls
- **Pydantic**: Data validation (included with FastAPI)

## Architecture

The backend follows a simple, clean architecture:

1. **FastAPI App**: Main application with CORS middleware
2. **API Endpoints**: RESTful endpoints for drug information
3. **OpenFDA Integration**: Async HTTP calls to FDA's drug database
4. **Error Handling**: Comprehensive error handling with proper HTTP responses
5. **Type Safety**: Full type hints for better code quality

## Development

### Running in Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-reload when code changes.

### Testing
You can test the API using:
- **curl** commands
- **Postman** or similar API testing tools
- **The React frontend** (make sure it's running on port 3000)

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in `main.py` or kill the process using port 8000
2. **CORS errors**: Ensure the frontend is running on the correct port (3000)
3. **OpenFDA API errors**: Check your internet connection and the OpenFDA API status

### Logs
The application provides detailed error logging for debugging API issues.
