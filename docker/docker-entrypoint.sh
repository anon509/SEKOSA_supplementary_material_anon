#!/bin/bash
set -e

echo "==================================="
echo "SEKOSA Docker Container"
echo "==================================="

# Start TypeDB server in background
echo "Starting TypeDB server..."
typedb server --server.address=0.0.0.0:1729 &
TYPEDB_PID=$!

# Wait for TypeDB to be ready
echo "Waiting for TypeDB to be ready..."
sleep 10

# Initialize database using Python script
echo "Initializing SEKOSA database..."
/opt/venv/bin/python /app/initialize_database.py

if [ $? -eq 0 ]; then
    echo "Database initialization completed successfully!"
else
    echo "Database initialization failed!"
    exit 1
fi

echo "==================================="
echo "TypeDB server is running on port 1729"
echo ""
echo "To run the demonstration:"
echo "  docker exec -it sekosa-container /opt/venv/bin/python /app/SEKOSA_demonstration.py"
echo ""
echo "Or enter the container:"
echo "  docker exec -it sekosa-container bash"
echo "==================================="

# Keep the container running
wait $TYPEDB_PID
