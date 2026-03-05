package com.finsight.dto;

import lombok.Builder;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@Builder
public class TransactionResponse {
    private Long id;
    private BigDecimal amount;
    private String type;
    private Long categoryId;
    private String categoryName;
    private String description;
    private LocalDate date;
    private LocalDateTime createdAt;
}
