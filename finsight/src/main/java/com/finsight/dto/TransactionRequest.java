package com.finsight.dto;

import com.finsight.model.Transaction;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDate;

@Data
public class TransactionRequest {
    @NotNull
    @Positive
    private BigDecimal amount;

    @NotNull
    private Transaction.TransactionType type;

    private Long categoryId;

    private String description;

    @NotNull
    private LocalDate date;
}
