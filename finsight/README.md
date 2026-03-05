# 💰 FinSight – Personal Finance Analytics Platform

<p align="center">
  <img src="https://img.shields.io/badge/Java-17-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white"/>
  <img src="https://img.shields.io/badge/Spring%20Boot-3.2.5-6DB33F?style=for-the-badge&logo=springboot&logoColor=white"/>
  <img src="https://img.shields.io/badge/MySQL-8.0-4479A1?style=for-the-badge&logo=mysql&logoColor=white"/>
  <img src="https://img.shields.io/badge/JWT-Auth-000000?style=for-the-badge&logo=jsonwebtokens&logoColor=white"/>
  <img src="https://img.shields.io/badge/Swagger-OpenAPI-85EA2D?style=for-the-badge&logo=swagger&logoColor=black"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
</p>

<p align="center">
  <b>A production-grade fintech backend system for personal finance management, analytics, and spending anomaly detection.</b>
</p>

---

## 📋 Overview

FinSight is a RESTful backend API that helps users manage their personal finances. It provides:

- 🔐 **Secure authentication** with JWT tokens and BCrypt password hashing
- 💳 **Transaction management** for income and expenses with categories
- 📊 **Budget tracking** with overspend alerts
- 📈 **Financial analytics**: monthly summaries, top spending categories, and 6-month trends
- 🔍 **Anomaly detection** using Z-score statistical analysis
- 📖 **Swagger UI** for interactive API documentation

---

## 🏗️ Architecture

```
src/main/java/com/finsight/
├── FinSightApplication.java        # Application entry point
├── config/
│   ├── SwaggerConfig.java          # OpenAPI / Swagger configuration with JWT auth
│   └── RedisConfig.java            # Optional Redis caching configuration
├── controller/
│   ├── AuthController.java         # Registration and login endpoints
│   ├── TransactionController.java  # CRUD + filtering for transactions
│   ├── CategoryController.java     # Category management
│   ├── BudgetController.java       # Budget creation and status tracking
│   └── AnalyticsController.java    # Financial insights and anomaly detection
├── service/
│   ├── AuthService.java            # User registration, login, JWT generation
│   ├── TransactionService.java     # Transaction business logic
│   ├── CategoryService.java        # Category business logic
│   ├── BudgetService.java          # Budget management and overspend detection
│   └── AnalyticsService.java       # Analytics calculations and anomaly detection
├── repository/
│   ├── UserRepository.java         # User data access
│   ├── TransactionRepository.java  # Transaction queries with JPQL
│   ├── CategoryRepository.java     # Category data access
│   └── BudgetRepository.java       # Budget data access
├── model/
│   ├── User.java                   # User entity (roles: USER, ADMIN)
│   ├── Transaction.java            # Transaction entity (INCOME / EXPENSE)
│   ├── Category.java               # Category entity
│   └── Budget.java                 # Monthly budget entity
├── dto/                            # Request/response DTOs (no entity exposure)
├── security/
│   ├── JwtTokenProvider.java       # JWT generation and validation (jjwt 0.11.5)
│   ├── JwtAuthenticationFilter.java # Per-request JWT filter
│   ├── CustomUserDetailsService.java # Spring Security user loading
│   └── SecurityConfig.java         # Security filter chain configuration
├── exception/
│   ├── GlobalExceptionHandler.java  # @RestControllerAdvice with proper HTTP codes
│   ├── ResourceNotFoundException.java
│   ├── BadRequestException.java
│   └── UnauthorizedException.java
└── util/
    └── SecurityUtils.java           # Helper to get current authenticated user
```

---

## 🛠️ Tech Stack

| Component           | Technology                      |
|---------------------|---------------------------------|
| Language            | Java 17                         |
| Framework           | Spring Boot 3.2.5               |
| Security            | Spring Security + JWT (jjwt 0.11.5) |
| ORM                 | Spring Data JPA / Hibernate     |
| Database            | MySQL 8.0 (or PostgreSQL)       |
| Build Tool          | Maven                           |
| Boilerplate         | Lombok                          |
| API Documentation   | springdoc-openapi 2.3.0 (Swagger UI) |
| Monitoring          | Spring Boot Actuator            |
| Containerization    | Docker + Docker Compose         |
| Caching (optional)  | Redis                           |

---

## 🗄️ Database Schema

```
users               transactions             categories            budgets
─────────────────   ───────────────────────  ──────────────────    ─────────────────────
id (PK)             id (PK)                  id (PK)               id (PK)
name                user_id (FK → users)     name                  user_id (FK → users)
email (unique)      amount                   type (INCOME/EXPENSE) category_id (FK)
password (BCrypt)   type (INCOME/EXPENSE)    user_id (FK → users)  monthly_limit
role                category_id (FK)                               month
created_at          description                                     year
                    date
                    created_at
```

---

## 🔌 API Endpoints

### Authentication
| Method | Endpoint               | Description        | Auth Required |
|--------|------------------------|--------------------|---------------|
| POST   | `/api/auth/register`   | Register new user  | No            |
| POST   | `/api/auth/login`      | Login, get JWT     | No            |

### Transactions
| Method | Endpoint                        | Description                         | Auth Required |
|--------|---------------------------------|-------------------------------------|---------------|
| POST   | `/api/transactions`             | Create a transaction                | Yes           |
| PUT    | `/api/transactions/{id}`        | Update a transaction                | Yes           |
| DELETE | `/api/transactions/{id}`        | Delete a transaction                | Yes           |
| GET    | `/api/transactions`             | Get transactions (paginated)        | Yes           |
| GET    | `/api/transactions/monthly`     | Get transactions by month/year      | Yes           |

### Categories
| Method | Endpoint           | Description          | Auth Required |
|--------|--------------------|----------------------|---------------|
| POST   | `/api/categories`  | Create a category    | Yes           |
| GET    | `/api/categories`  | List all categories  | Yes           |

### Budgets
| Method | Endpoint                | Description                    | Auth Required |
|--------|-------------------------|--------------------------------|---------------|
| POST   | `/api/budgets`          | Create a monthly budget        | Yes           |
| GET    | `/api/budgets`          | List budgets by month/year     | Yes           |
| GET    | `/api/budgets/status`   | Budget vs actual spending      | Yes           |

### Analytics
| Method | Endpoint                              | Description                          | Auth Required |
|--------|---------------------------------------|--------------------------------------|---------------|
| GET    | `/api/analytics/monthly-summary`      | Income, expense, savings summary     | Yes           |
| GET    | `/api/analytics/top-categories`       | Top N spending categories            | Yes           |
| GET    | `/api/analytics/spending-trends`      | Last 6 months income/expense trends  | Yes           |
| GET    | `/api/analytics/anomaly-detection`    | Detect abnormal transactions (Z-score) | Yes         |

---

## 🚀 Local Setup

### Prerequisites

- Java 17+
- Maven 3.9+
- MySQL 8.0 (or Docker)

---

### Option A: Run with Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/shashankbaswa007/melanoma-skin-cancer-detection-model.git
cd melanoma-skin-cancer-detection-model/finsight

# Copy the example env file and fill in your secrets
cp .env.example .env   # then edit .env with your DB credentials and JWT secret

# Start MySQL + App
docker-compose up --build
```

The application will be available at `http://localhost:8080`.

---

### Option B: Run Locally

#### 1. Install Java 17

```bash
# Ubuntu/Debian
sudo apt install openjdk-17-jdk

# macOS (Homebrew)
brew install openjdk@17

# Verify
java --version
```

#### 2. Set up MySQL

```bash
# Install MySQL
sudo apt install mysql-server   # Ubuntu
brew install mysql              # macOS

# Start MySQL and create database
mysql -u root -p
CREATE DATABASE finsight;
```

#### 3. Configure `application.properties`

Edit `src/main/resources/application.properties`:

```properties
spring.datasource.url=jdbc:mysql://localhost:3306/finsight?createDatabaseIfNotExist=true&useSSL=false&allowPublicKeyRetrieval=true
spring.datasource.username=your_mysql_user
spring.datasource.password=your_mysql_password

# Change JWT secret to a strong random string in production
jwt.secret=your-production-secret-key-minimum-32-characters
jwt.expiration=86400000
```

#### 4. Build and Run

```bash
cd finsight
mvn clean package -DskipTests
java -jar target/finsight-1.0.0.jar
```

---

## 📖 API Documentation (Swagger UI)

Once the application is running, open:

**[http://localhost:8080/swagger-ui.html](http://localhost:8080/swagger-ui.html)**

To test secured endpoints:
1. Call `POST /api/auth/register` to create an account
2. Call `POST /api/auth/login` to receive a JWT token
3. Click **Authorize** in Swagger UI and enter: `<your_token>`
4. All subsequent requests will use the JWT automatically

---

## 📦 Sample API Requests

### Register a User

```bash
curl -X POST http://localhost:8080/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "password": "secret123"
  }'
```

### Login

```bash
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john@example.com",
    "password": "secret123"
  }'
# Response: { "token": "eyJhbGci...", "type": "Bearer", "email": "john@example.com", "name": "John Doe" }
```

### Add a Transaction

```bash
TOKEN="eyJhbGci..."
curl -X POST http://localhost:8080/api/transactions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 150.00,
    "type": "EXPENSE",
    "categoryId": 1,
    "description": "Grocery shopping",
    "date": "2024-01-15"
  }'
```

### Get Monthly Summary

```bash
curl -X GET "http://localhost:8080/api/analytics/monthly-summary?month=1&year=2024" \
  -H "Authorization: Bearer $TOKEN"
# Response: { "month": 1, "year": 2024, "totalIncome": 5000.00, "totalExpense": 3200.00,
#             "netSavings": 1800.00, "savingsRate": 36.0 }
```

### Detect Spending Anomalies

```bash
curl -X GET http://localhost:8080/api/analytics/anomaly-detection \
  -H "Authorization: Bearer $TOKEN"
# Returns transactions with Z-score > 2.0 (significantly above average spending)
```

---

## 🔒 Security

- All passwords are hashed using **BCrypt** (never stored in plaintext)
- API endpoints are protected via **JWT Bearer token** authentication
- JWT tokens expire after **24 hours** (configurable via `jwt.expiration`)
- Stateless session management (no server-side sessions)
- Input validation on all request bodies using Spring Validation annotations
- Role-based authorization support (`USER` / `ADMIN`)

---

## 📊 Analytics Features

### Monthly Summary
Returns total income, total expense, net savings, and savings rate for a given month.

### Top Spending Categories
Returns the top N categories by total spending for a given month, with percentage breakdown.

### Spending Trends
Returns month-by-month income/expense data for the last 6 months.

### Anomaly Detection (Z-Score Method)
Flags transactions where:
```
Z-score = (amount - mean_expense) / stddev_expense > 2.0
```
These represent expenses significantly higher than the user's historical average.

---

## 🏥 Health & Monitoring

Spring Boot Actuator endpoints:

```bash
# Health check
curl http://localhost:8080/actuator/health

# Application info
curl http://localhost:8080/actuator/info

# Metrics
curl http://localhost:8080/actuator/metrics
```

---

## 🐳 Docker

```bash
# Build image only
docker build -t finsight .

# Run with Docker Compose (MySQL + App)
docker-compose up -d

# Stop everything
docker-compose down
```

---

## ⚙️ Configuration Reference

| Property                          | Default                    | Description                    |
|-----------------------------------|----------------------------|--------------------------------|
| `spring.datasource.url`           | MySQL localhost:3306        | Database connection URL        |
| `spring.datasource.username`      | `root`                     | Database username              |
| `spring.datasource.password`      | `password`                 | Database password              |
| `jwt.secret`                      | (see application.properties) | JWT signing secret (min 32 chars) |
| `jwt.expiration`                  | `86400000` (24h)           | Token expiry in milliseconds   |
| `spring.jpa.hibernate.ddl-auto`   | `update`                   | Schema management strategy     |

---

## 📁 Project Structure

```
finsight/
├── pom.xml                         # Maven dependencies
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml              # MySQL + App orchestration
├── README.md
└── src/
    └── main/
        ├── java/com/finsight/      # Java source files (43 classes)
        └── resources/
            └── application.properties
```
