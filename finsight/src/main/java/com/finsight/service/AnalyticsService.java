package com.finsight.service;

import com.finsight.dto.MonthlySummaryResponse;
import com.finsight.dto.SpendingTrendResponse;
import com.finsight.dto.TopCategoryResponse;
import com.finsight.dto.TransactionResponse;
import com.finsight.model.Transaction;
import com.finsight.model.User;
import com.finsight.repository.TransactionRepository;
import com.finsight.util.SecurityUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.time.Month;
import java.time.format.TextStyle;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class AnalyticsService {

    private final TransactionRepository transactionRepository;
    private final SecurityUtils securityUtils;

    public MonthlySummaryResponse getMonthlySummary(int month, int year) {
        User currentUser = securityUtils.getCurrentUser();
        Long userId = currentUser.getId();

        BigDecimal totalIncome = transactionRepository.sumByUserIdAndTypeAndMonth(
                userId, Transaction.TransactionType.INCOME, year, month);
        BigDecimal totalExpense = transactionRepository.sumByUserIdAndTypeAndMonth(
                userId, Transaction.TransactionType.EXPENSE, year, month);

        totalIncome = totalIncome != null ? totalIncome : BigDecimal.ZERO;
        totalExpense = totalExpense != null ? totalExpense : BigDecimal.ZERO;

        BigDecimal netSavings = totalIncome.subtract(totalExpense);
        double savingsRate = totalIncome.compareTo(BigDecimal.ZERO) > 0
                ? netSavings.divide(totalIncome, 4, RoundingMode.HALF_UP).doubleValue() * 100
                : 0;

        return MonthlySummaryResponse.builder()
                .month(month)
                .year(year)
                .totalIncome(totalIncome)
                .totalExpense(totalExpense)
                .netSavings(netSavings)
                .savingsRate(savingsRate)
                .build();
    }

    public List<TopCategoryResponse> getTopCategories(int month, int year, int limit) {
        User currentUser = securityUtils.getCurrentUser();
        List<Object[]> raw = transactionRepository.findTopSpendingCategories(currentUser.getId(), year, month);

        BigDecimal totalExpense = raw.stream()
                .map(row -> (BigDecimal) row[1])
                .reduce(BigDecimal.ZERO, BigDecimal::add);

        return raw.stream()
                .limit(limit)
                .map(row -> {
                    String categoryName = (String) row[0];
                    BigDecimal amount = (BigDecimal) row[1];
                    double percentage = totalExpense.compareTo(BigDecimal.ZERO) > 0
                            ? amount.divide(totalExpense, 4, RoundingMode.HALF_UP).doubleValue() * 100
                            : 0;
                    return TopCategoryResponse.builder()
                            .categoryName(categoryName)
                            .totalAmount(amount)
                            .percentage(percentage)
                            .build();
                })
                .collect(Collectors.toList());
    }

    public List<SpendingTrendResponse> getSpendingTrends() {
        User currentUser = securityUtils.getCurrentUser();
        Long userId = currentUser.getId();

        List<SpendingTrendResponse> trends = new ArrayList<>();
        LocalDate now = LocalDate.now();

        for (int i = 5; i >= 0; i--) {
            LocalDate date = now.minusMonths(i);
            int month = date.getMonthValue();
            int year = date.getYear();

            BigDecimal income = transactionRepository.sumByUserIdAndTypeAndMonth(userId, Transaction.TransactionType.INCOME, year, month);
            BigDecimal expense = transactionRepository.sumByUserIdAndTypeAndMonth(userId, Transaction.TransactionType.EXPENSE, year, month);

            income = income != null ? income : BigDecimal.ZERO;
            expense = expense != null ? expense : BigDecimal.ZERO;

            String monthLabel = Month.of(month).getDisplayName(TextStyle.SHORT, Locale.ENGLISH) + " " + year;

            trends.add(SpendingTrendResponse.builder()
                    .month(month)
                    .year(year)
                    .monthLabel(monthLabel)
                    .totalIncome(income)
                    .totalExpense(expense)
                    .netSavings(income.subtract(expense))
                    .build());
        }

        return trends;
    }

    public List<TransactionResponse> detectAnomalies() {
        User currentUser = securityUtils.getCurrentUser();
        Long userId = currentUser.getId();

        BigDecimal mean = transactionRepository.findAverageExpenseAmount(userId);
        BigDecimal stdDev = transactionRepository.findStdDevExpenseAmount(userId);

        if (mean == null || stdDev == null || stdDev.compareTo(BigDecimal.ZERO) == 0) {
            return List.of();
        }

        List<Transaction> allExpenses = transactionRepository.findExpensesByUserId(userId);

        return allExpenses.stream()
                .filter(t -> {
                    BigDecimal zScore = t.getAmount().subtract(mean)
                            .divide(stdDev, 4, RoundingMode.HALF_UP);
                    return zScore.doubleValue() > 2.0;
                })
                .map(t -> TransactionResponse.builder()
                        .id(t.getId())
                        .amount(t.getAmount())
                        .type(t.getType().name())
                        .categoryId(t.getCategory() != null ? t.getCategory().getId() : null)
                        .categoryName(t.getCategory() != null ? t.getCategory().getName() : null)
                        .description(t.getDescription())
                        .date(t.getDate())
                        .createdAt(t.getCreatedAt())
                        .build())
                .collect(Collectors.toList());
    }
}
