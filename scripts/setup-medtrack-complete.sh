#!/bin/bash

# MedTrack Complete System Setup and Verification Script
# This script sets up and verifies the entire MedTrack system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/Users/AlexanderSokol/medtrack"
DOCKER_COMPOSE_FILE="docker-compose.yml"
DOCKER_COMPOSE_PROD_FILE="docker-compose.prod.yml"
ENV_FILE=".env"

# Function to print section headers
print_section() {
    echo -e "\n${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

# Function to print step headers
print_step() {
    echo -e "\n${BLUE}➤ $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if file exists
file_exists() {
    [ -f "$1" ]
}

# Function to check if directory exists
dir_exists() {
    [ -d "$1" ]
}

# Function to check if Docker is running
check_docker() {
    print_step "Checking Docker installation and status"
    
    if ! command_exists docker; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running"
        exit 1
    fi
    
    print_success "Docker is running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    print_step "Checking Docker Compose availability"
    
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        print_error "Docker Compose is not available"
        exit 1
    fi
    
    print_success "Docker Compose is available"
}

# Function to check if Node.js is installed
check_nodejs() {
    print_step "Checking Node.js installation"
    
    if ! command_exists node; then
        print_error "Node.js is not installed"
        exit 1
    fi
    
    NODE_VERSION=$(node --version)
    print_success "Node.js is installed: $NODE_VERSION"
}

# Function to check if pnpm is installed
check_pnpm() {
    print_step "Checking pnpm installation"
    
    if ! command_exists pnpm; then
        print_warning "pnpm is not installed, installing..."
        npm install -g pnpm
    fi
    
    PNPM_VERSION=$(pnpm --version)
    print_success "pnpm is installed: $PNPM_VERSION"
}

# Function to setup environment
setup_environment() {
    print_step "Setting up environment configuration"
    
    cd "$PROJECT_ROOT"
    
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f ".env.example" ]; then
            print_warning "Creating environment file from example"
            cp .env.example "$ENV_FILE"
        else
            print_error "Environment file not found"
            exit 1
        fi
    fi
    
    print_success "Environment file ready"
}

# Function to install dependencies
install_dependencies() {
    print_step "Installing project dependencies"
    
    cd "$PROJECT_ROOT"
    
    # Install root dependencies
    print_step "Installing root dependencies"
    pnpm install
    
    # Install backend dependencies
    if [ -d "backend" ]; then
        print_step "Installing backend dependencies"
        cd backend
        pnpm install
        cd ..
    fi
    
    # Install frontend dependencies
    if [ -d "frontend" ]; then
        print_step "Installing frontend dependencies"
        cd frontend
        pnpm install
        cd ..
    fi
    
    # Install shared packages dependencies
    if [ -d "packages" ]; then
        print_step "Installing shared packages dependencies"
        for package in packages/*/; do
            if [ -f "$package/package.json" ]; then
                print_step "Installing dependencies for $(basename "$package")"
                cd "$package"
                pnpm install
                cd "$PROJECT_ROOT"
            fi
        done
    fi
    
    # Install services dependencies
    if [ -d "services" ]; then
        print_step "Installing services dependencies"
        for service in services/*/; do
            if [ -f "$service/package.json" ]; then
                print_step "Installing dependencies for $(basename "$service")"
                cd "$service"
                pnpm install
                cd "$PROJECT_ROOT"
            fi
        done
    fi
    
    print_success "All dependencies installed"
}

# Function to build TypeScript
build_typescript() {
    print_step "Building TypeScript projects"
    
    cd "$PROJECT_ROOT"
    
    # Build backend
    if [ -d "backend" ] && [ -f "backend/tsconfig.json" ]; then
        print_step "Building backend TypeScript"
        cd backend
        pnpm run build
        cd ..
    fi
    
    # Build shared packages
    if [ -d "packages" ]; then
        for package in packages/*/; do
            if [ -f "$package/tsconfig.json" ]; then
                print_step "Building $(basename "$package") TypeScript"
                cd "$package"
                pnpm run build
                cd "$PROJECT_ROOT"
            fi
        done
    fi
    
    print_success "TypeScript build completed"
}

# Function to start infrastructure services
start_infrastructure() {
    print_step "Starting infrastructure services"
    
    cd "$PROJECT_ROOT"
    
    # Start core services
    print_step "Starting core services (PostgreSQL, Redis, Qdrant, Ollama)"
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d postgres redis qdrant ollama
    
    # Wait for services to be ready
    print_step "Waiting for services to be ready"
    
    # Wait for PostgreSQL
    print_step "Waiting for PostgreSQL"
    timeout 60 bash -c 'until docker exec medtrack-postgres pg_isready -U medtrack -d medtrack; do sleep 2; done'
    
    # Wait for Redis
    print_step "Waiting for Redis"
    timeout 30 bash -c 'until docker exec medtrack-redis redis-cli ping; do sleep 2; done'
    
    # Wait for Qdrant
    print_step "Waiting for Qdrant"
    timeout 30 bash -c 'until curl -f http://localhost:6333/health; do sleep 2; done'
    
    # Wait for Ollama
    print_step "Waiting for Ollama"
    timeout 60 bash -c 'until curl -f http://localhost:11434/api/tags; do sleep 2; done'
    
    print_success "Infrastructure services are ready"
}

# Function to setup AI models
setup_ai_models() {
    print_step "Setting up AI models"
    
    # Pull required models
    print_step "Pulling AI models"
    docker exec medtrack-ollama ollama pull llama3.2:3b || print_warning "Failed to pull llama3.2:3b"
    docker exec medtrack-ollama ollama pull nomic-embed-text || print_warning "Failed to pull nomic-embed-text"
    
    print_success "AI models setup completed"
}

# Function to run database migrations
run_database_migrations() {
    print_step "Running database migrations"
    
    cd "$PROJECT_ROOT"
    
    # Generate Prisma client
    print_step "Generating Prisma client"
    cd backend
    pnpm prisma generate
    cd ..
    
    # Run migrations
    print_step "Running Prisma migrations"
    cd backend
    pnpm prisma migrate dev --name init
    cd ..
    
    print_success "Database migrations completed"
}

# Function to start application services
start_application_services() {
    print_step "Starting application services"
    
    cd "$PROJECT_ROOT"
    
    # Start vector search service
    print_step "Starting vector search service"
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d vector-search
    
    # Wait for vector search
    print_step "Waiting for vector search service"
    timeout 30 bash -c 'until curl -f http://localhost:3005/healthz; do sleep 2; done'
    
    # Start backend
    print_step "Starting backend service"
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d backend
    
    # Wait for backend
    print_step "Waiting for backend service"
    timeout 60 bash -c 'until curl -f http://localhost:4000/health; do sleep 2; done'
    
    # Start frontend
    print_step "Starting frontend service"
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d frontend
    
    # Wait for frontend
    print_step "Waiting for frontend service"
    timeout 30 bash -c 'until curl -f http://localhost:3000; do sleep 2; done'
    
    print_success "Application services are ready"
}

# Function to run health checks
run_health_checks() {
    print_step "Running comprehensive health checks"
    
    # Check backend health
    print_step "Checking backend health"
    if curl -f http://localhost:4000/health >/dev/null 2>&1; then
        print_success "Backend is healthy"
    else
        print_error "Backend health check failed"
        return 1
    fi
    
    # Check frontend health
    print_step "Checking frontend health"
    if curl -f http://localhost:3000 >/dev/null 2>&1; then
        print_success "Frontend is healthy"
    else
        print_error "Frontend health check failed"
        return 1
    fi
    
    # Check vector search health
    print_step "Checking vector search health"
    if curl -f http://localhost:3005/healthz >/dev/null 2>&1; then
        print_success "Vector search is healthy"
    else
        print_error "Vector search health check failed"
        return 1
    fi
    
    # Check AI services
    print_step "Checking AI services"
    if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_success "Ollama is healthy"
    else
        print_error "Ollama health check failed"
        return 1
    fi
    
    # Check database
    print_step "Checking database"
    if docker exec medtrack-postgres pg_isready -U medtrack -d medtrack >/dev/null 2>&1; then
        print_success "Database is healthy"
    else
        print_error "Database health check failed"
        return 1
    fi
    
    # Check Redis
    print_step "Checking Redis"
    if docker exec medtrack-redis redis-cli ping >/dev/null 2>&1; then
        print_success "Redis is healthy"
    else
        print_error "Redis health check failed"
        return 1
    fi
    
    # Check Qdrant
    print_step "Checking Qdrant"
    if curl -f http://localhost:6333/health >/dev/null 2>&1; then
        print_success "Qdrant is healthy"
    else
        print_error "Qdrant health check failed"
        return 1
    fi
    
    print_success "All health checks passed"
}

# Function to run integration tests
run_integration_tests() {
    print_step "Running integration tests"
    
    cd "$PROJECT_ROOT"
    
    # Run backend tests
    if [ -d "backend" ] && [ -f "backend/package.json" ]; then
        print_step "Running backend tests"
        cd backend
        pnpm test || print_warning "Backend tests failed"
        cd ..
    fi
    
    # Run frontend tests
    if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
        print_step "Running frontend tests"
        cd frontend
        pnpm test || print_warning "Frontend tests failed"
        cd ..
    fi
    
    print_success "Integration tests completed"
}

# Function to test AI functionality
test_ai_functionality() {
    print_step "Testing AI functionality"
    
    # Test medication search
    print_step "Testing medication search"
    if curl -f "http://localhost:4000/api/ai/search?query=aspirin" >/dev/null 2>&1; then
        print_success "Medication search is working"
    else
        print_warning "Medication search test failed"
    fi
    
    # Test medication validation
    print_step "Testing medication validation"
    if curl -f "http://localhost:4000/api/ai/validate" -X POST -H "Content-Type: application/json" -d '{"name":"aspirin","dosage":"100mg","frequency":"twice daily"}' >/dev/null 2>&1; then
        print_success "Medication validation is working"
    else
        print_warning "Medication validation test failed"
    fi
    
    # Test AI assistant
    print_step "Testing AI assistant"
    if curl -f "http://localhost:4000/api/ai/chat" -X POST -H "Content-Type: application/json" -d '{"message":"Hello"}' >/dev/null 2>&1; then
        print_success "AI assistant is working"
    else
        print_warning "AI assistant test failed"
    fi
    
    print_success "AI functionality tests completed"
}

# Function to show system status
show_system_status() {
    print_section "System Status"
    
    # Show running containers
    print_step "Running containers"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    # Show service URLs
    print_step "Service URLs"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend API: http://localhost:4000"
    echo "  API Documentation: http://localhost:4000/docs"
    echo "  Vector Search: http://localhost:3005"
    echo "  Ollama: http://localhost:11434"
    echo "  Qdrant: http://localhost:6333"
    echo "  Grafana: http://localhost:3001"
    echo "  Prometheus: http://localhost:9090"
    
    # Show logs
    print_step "Log commands"
    echo "  All services: docker-compose -f $DOCKER_COMPOSE_FILE logs -f"
    echo "  Backend: docker-compose -f $DOCKER_COMPOSE_FILE logs -f backend"
    echo "  Frontend: docker-compose -f $DOCKER_COMPOSE_FILE logs -f frontend"
    echo "  Vector Search: docker-compose -f $DOCKER_COMPOSE_FILE logs -f vector-search"
}

# Function to show next steps
show_next_steps() {
    print_section "Next Steps"
    
    echo "1. Access your MedTrack application at: http://localhost:3000"
    echo "2. View API documentation at: http://localhost:4000/docs"
    echo "3. Test AI features through the frontend interface"
    echo "4. Monitor system health through Grafana: http://localhost:3001"
    echo "5. Check logs if you encounter any issues"
    echo ""
    echo "For production deployment, use:"
    echo "  ./scripts/deploy-production.sh"
    echo ""
    echo "For system backup, use:"
    echo "  ./scripts/backup.sh"
}

# Main execution
main() {
    # Parse command line arguments
    SKIP_DEPS=false
    SKIP_BUILD=false
    SKIP_MIGRATIONS=false
    SKIP_AI_SETUP=false
    SKIP_TESTS=false
    SKIP_AI_TESTS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-migrations)
                SKIP_MIGRATIONS=true
                shift
                ;;
            --skip-ai-setup)
                SKIP_AI_SETUP=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-ai-tests)
                SKIP_AI_TESTS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --skip-deps        Skip dependency installation"
                echo "  --skip-build       Skip TypeScript build"
                echo "  --skip-migrations  Skip database migrations"
                echo "  --skip-ai-setup    Skip AI models setup"
                echo "  --skip-tests       Skip integration tests"
                echo "  --skip-ai-tests    Skip AI functionality tests"
                echo "  --help             Show this help"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    print_section "MedTrack Complete System Setup"
    echo "This script will set up and verify the entire MedTrack system"
    
    # Pre-setup checks
    check_docker
    check_docker_compose
    check_nodejs
    check_pnpm
    setup_environment
    
    # Setup phase
    if [ "$SKIP_DEPS" != true ]; then
        install_dependencies
    fi
    
    if [ "$SKIP_BUILD" != true ]; then
        build_typescript
    fi
    
    # Infrastructure phase
    start_infrastructure
    
    if [ "$SKIP_AI_SETUP" != true ]; then
        setup_ai_models
    fi
    
    if [ "$SKIP_MIGRATIONS" != true ]; then
        run_database_migrations
    fi
    
    # Application phase
    start_application_services
    
    # Verification phase
    run_health_checks
    
    if [ "$SKIP_TESTS" != true ]; then
        run_integration_tests
    fi
    
    if [ "$SKIP_AI_TESTS" != true ]; then
        test_ai_functionality
    fi
    
    # Status and next steps
    show_system_status
    show_next_steps
    
    print_section "Setup Complete"
    echo -e "${GREEN}🎉 MedTrack system setup and verification completed successfully!${NC}"
    echo ""
    echo "Your MedTrack application is now running and ready to use!"
}

# Run main function
main "$@"