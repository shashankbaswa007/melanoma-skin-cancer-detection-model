package com.finsight.config;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Configuration;

/**
 * Redis configuration stub. Enable by uncommenting Redis properties in application.properties
 * and adding @EnableCaching to this class or FinSightApplication.
 */
@Configuration
@ConditionalOnProperty(name = "spring.cache.type", havingValue = "redis")
public class RedisConfig {
    // Redis configuration is auto-configured by Spring Boot when spring-boot-starter-data-redis is on the classpath
    // and spring.data.redis.host / port are configured.
    // Add custom RedisTemplate or CacheManager beans here if needed.
}
