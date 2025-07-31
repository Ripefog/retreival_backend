#!/bin/bash

# Video Retrieval Backend Startup Script
# Usage: ./start.sh [dev|prod|docker]

set -e

echo "🚀 Starting Video Retrieval Backend..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3 first."
        exit 1
    fi
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p logs
    print_success "Directories created successfully"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Copy environment file
setup_environment() {
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            print_status "Creating .env file from template..."
            cp env.example .env
            print_warning "Please update .env file with your configuration"
        else
            print_warning "No .env file found. Please create one manually."
        fi
    fi
}

# Start development mode
start_dev() {
    print_status "Starting development environment..."
    
    check_python
    create_directories
    setup_environment
    install_dependencies
    
    print_status "Starting backend server..."
    cd app
    python3 main.py
}

# Start production mode
start_prod() {
    print_status "Starting production environment..."
    
    check_python
    create_directories
    setup_environment
    install_dependencies
    
    print_status "Starting production server..."
    cd app
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
}

# Start with Docker
start_docker() {
    print_status "Starting with Docker..."
    
    check_docker
    create_directories
    setup_environment
    
    print_status "Building and starting containers..."
    docker-compose up -d
    
    print_success "Docker containers started successfully!"
    print_status "Backend API: http://localhost:8000"
    print_status "API Docs: http://localhost:8000/docs"
    
    print_status "Checking service health..."
    sleep 10
    
    # Health checks
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Backend API is healthy"
    else
        print_warning "Backend API health check failed"
    fi
    
    print_status "Use 'docker-compose logs -f' to view logs"
    print_status "Use 'docker-compose down' to stop services"
}

# Start production with Docker
start_docker_prod() {
    print_status "Starting production with Docker..."
    
    check_docker
    create_directories
    setup_environment
    
    print_status "Building and starting production containers..."
    docker-compose --profile production up -d
    
    print_success "Production containers started successfully!"
    print_status "Backend API: http://localhost:8000"
    print_status "Nginx: http://localhost:80"
    print_status "Redis: localhost:6379"
    
    print_status "Use 'docker-compose --profile production logs -f' to view logs"
    print_status "Use 'docker-compose --profile production down' to stop services"
}

# Show help
show_help() {
    echo "Usage: $0 [dev|prod|docker|docker-prod|help]"
    echo ""
    echo "Options:"
    echo "  dev          Start development environment"
    echo "  prod         Start production environment"
    echo "  docker       Start with Docker (development)"
    echo "  docker-prod  Start with Docker (production)"
    echo "  help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev"
    echo "  $0 prod"
    echo "  $0 docker"
    echo "  $0 docker-prod"
}

# Main script
case "${1:-help}" in
    "dev")
        start_dev
        ;;
    "prod")
        start_prod
        ;;
    "docker")
        start_docker
        ;;
    "docker-prod")
        start_docker_prod
        ;;
    "help"|*)
        show_help
        ;;
esac 