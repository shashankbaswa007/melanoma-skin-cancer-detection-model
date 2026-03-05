package com.finsight.controller;

import com.finsight.dto.BudgetRequest;
import com.finsight.dto.BudgetResponse;
import com.finsight.dto.BudgetStatusResponse;
import com.finsight.service.BudgetService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/budgets")
@RequiredArgsConstructor
@Tag(name = "Budgets", description = "Budget management endpoints")
public class BudgetController {

    private final BudgetService budgetService;

    @PostMapping
    @Operation(summary = "Create a new budget")
    public ResponseEntity<BudgetResponse> createBudget(@Valid @RequestBody BudgetRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED).body(budgetService.createBudget(request));
    }

    @GetMapping
    @Operation(summary = "Get budgets (optionally filtered by month and year)")
    public ResponseEntity<List<BudgetResponse>> getBudgets(
            @RequestParam(required = false) Integer month,
            @RequestParam(required = false) Integer year) {
        return ResponseEntity.ok(budgetService.getBudgets(month, year));
    }

    @GetMapping("/status")
    @Operation(summary = "Get budget status with spending comparison for a given month/year")
    public ResponseEntity<List<BudgetStatusResponse>> getBudgetStatus(
            @RequestParam int month,
            @RequestParam int year) {
        return ResponseEntity.ok(budgetService.getBudgetStatus(month, year));
    }
}
