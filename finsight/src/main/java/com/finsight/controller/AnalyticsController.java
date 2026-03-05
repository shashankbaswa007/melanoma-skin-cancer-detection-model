package com.finsight.controller;

import com.finsight.dto.MonthlySummaryResponse;
import com.finsight.dto.SpendingTrendResponse;
import com.finsight.dto.TopCategoryResponse;
import com.finsight.dto.TransactionResponse;
import com.finsight.service.AnalyticsService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/analytics")
@RequiredArgsConstructor
@Tag(name = "Analytics", description = "Financial analytics endpoints")
public class AnalyticsController {

    private final AnalyticsService analyticsService;

    @GetMapping("/monthly-summary")
    @Operation(summary = "Get monthly income/expense summary")
    public ResponseEntity<MonthlySummaryResponse> getMonthlySummary(
            @RequestParam int month,
            @RequestParam int year) {
        return ResponseEntity.ok(analyticsService.getMonthlySummary(month, year));
    }

    @GetMapping("/top-categories")
    @Operation(summary = "Get top spending categories for a month")
    public ResponseEntity<List<TopCategoryResponse>> getTopCategories(
            @RequestParam int month,
            @RequestParam int year,
            @RequestParam(defaultValue = "5") int limit) {
        return ResponseEntity.ok(analyticsService.getTopCategories(month, year, limit));
    }

    @GetMapping("/spending-trends")
    @Operation(summary = "Get spending trends for the last 6 months")
    public ResponseEntity<List<SpendingTrendResponse>> getSpendingTrends() {
        return ResponseEntity.ok(analyticsService.getSpendingTrends());
    }

    @GetMapping("/anomaly-detection")
    @Operation(summary = "Detect anomalous transactions using Z-score method")
    public ResponseEntity<List<TransactionResponse>> detectAnomalies() {
        return ResponseEntity.ok(analyticsService.detectAnomalies());
    }
}
