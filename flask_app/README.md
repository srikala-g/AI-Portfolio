# Flask Hello World App

A basic Flask application that displays "Hello, World!" when accessed.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Run the Flask application:
```bash
python app.py
```

The app will start on `http://localhost:5000`. Open your browser and navigate to this URL to see "Hello, World!" displayed.

## Features

- Simple "Hello, World!" route at the root path (`/`)
- Debug mode enabled for development
- Runs on port 5000 by default

