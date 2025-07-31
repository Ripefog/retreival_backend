# Video Retrieval Backend Makefile

.PHONY: help install dev prod docker docker-prod test clean

# Default target
help:
	@echo "Video Retrieval Backend - Available commands:"
	@echo ""
	@echo "  install     - Install Python dependencies"
	@echo "  dev         - Start development server"
	@echo "  prod        - Start production server"
	@echo "  docker      - Start with Docker (development)"
	@echo "  docker-prod - Start with Docker (production)"
	@echo "  test        - Run API tests"
	@echo "  clean       - Clean up generated files"
	@echo "  logs        - View logs"
	@echo "  stop        - Stop all services"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Development mode
dev:
	@echo "🚀 Starting development server..."
	cd app && python main.py

# Production mode
prod:
	@echo "🚀 Starting production server..."
	cd app && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker development
docker:
	@echo "🐳 Starting with Docker (development)..."
	docker-compose up -d
	@echo "✅ Docker containers started"
	@echo "📊 API: http://localhost:8000"
	@echo "📚 Docs: http://localhost:8000/docs"

# Docker production
docker-prod:
	@echo "🐳 Starting with Docker (production)..."
	docker-compose --profile production up -d
	@echo "✅ Production containers started"
	@echo "📊 API: http://localhost:8000"
	@echo "🌐 Nginx: http://localhost:80"

# Run tests
test:
	@echo "🧪 Running API tests..."
	python test_api.py

# View logs
logs:
	@echo "📋 Viewing logs..."
	docker-compose logs -f

# Stop services
stop:
	@echo "🛑 Stopping services..."
	docker-compose down
	@echo "✅ Services stopped"

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage
	@echo "✅ Cleanup completed"

# Health check
health:
	@echo "🏥 Checking health..."
	curl -f http://localhost:8000/health || echo "❌ Health check failed"

# Format code
format:
	@echo "🎨 Formatting code..."
	black app/
	isort app/

# Lint code
lint:
	@echo "🔍 Linting code..."
	flake8 app/
	black --check app/
	isort --check-only app/

# Setup environment
setup:
	@echo "⚙️ Setting up environment..."
	@if [ ! -f .env ]; then \
		if [ -f env.example ]; then \
			cp env.example .env; \
			echo "✅ Created .env from template"; \
		else \
			echo "⚠️ No env.example found"; \
		fi; \
	fi
	mkdir -p logs
	@echo "✅ Environment setup completed" 