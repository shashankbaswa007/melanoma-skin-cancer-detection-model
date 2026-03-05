package com.finsight.controller;

import com.finsight.dto.TransactionRequest;
import com.finsight.dto.TransactionResponse;
import com.finsight.service.TransactionService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/transactions")
@RequiredArgsConstructor
@Tag(name = "Transactions", description = "Transaction management endpoints")
public class TransactionController {

    private final TransactionService transactionService;

    @PostMapping
    @Operation(summary = "Create a new transaction")
    public ResponseEntity<TransactionResponse> createTransaction(@Valid @RequestBody TransactionRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED).body(transactionService.createTransaction(request));
    }

    @PutMapping("/{id}")
    @Operation(summary = "Update an existing transaction")
    public ResponseEntity<TransactionResponse> updateTransaction(
            @PathVariable Long id,
            @Valid @RequestBody TransactionRequest request) {
        return ResponseEntity.ok(transactionService.updateTransaction(id, request));
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "Delete a transaction")
    public ResponseEntity<Void> deleteTransaction(@PathVariable Long id) {
        transactionService.deleteTransaction(id);
        return ResponseEntity.noContent().build();
    }

    @GetMapping
    @Operation(summary = "Get all transactions with optional filters and pagination")
    public ResponseEntity<Page<TransactionResponse>> getTransactions(
            @RequestParam(required = false) Long categoryId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        Pageable pageable = PageRequest.of(page, size);
        return ResponseEntity.ok(transactionService.getTransactions(categoryId, pageable));
    }

    @GetMapping("/monthly")
    @Operation(summary = "Get transactions for a specific month and year")
    public ResponseEntity<List<TransactionResponse>> getMonthlyTransactions(
            @RequestParam int month,
            @RequestParam int year) {
        return ResponseEntity.ok(transactionService.getMonthlyTransactions(month, year));
    }
}
