package com.finsight.service;

import com.finsight.dto.BudgetRequest;
import com.finsight.dto.BudgetResponse;
import com.finsight.dto.BudgetStatusResponse;
import com.finsight.exception.BadRequestException;
import com.finsight.exception.ResourceNotFoundException;
import com.finsight.model.Budget;
import com.finsight.model.Category;
import com.finsight.model.Transaction;
import com.finsight.model.User;
import com.finsight.repository.BudgetRepository;
import com.finsight.repository.CategoryRepository;
import com.finsight.repository.TransactionRepository;
import com.finsight.util.SecurityUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class BudgetService {

    private final BudgetRepository budgetRepository;
    private final CategoryRepository categoryRepository;
    private final TransactionRepository transactionRepository;
    private final SecurityUtils securityUtils;

    public BudgetResponse createBudget(BudgetRequest request) {
        User currentUser = securityUtils.getCurrentUser();

        Category category = categoryRepository.findById(request.getCategoryId())
                .orElseThrow(() -> new ResourceNotFoundException("Category not found with id: " + request.getCategoryId()));

        if (budgetRepository.findByUserIdAndCategoryIdAndMonthAndYear(
                currentUser.getId(), request.getCategoryId(), request.getMonth(), request.getYear()).isPresent()) {
            throw new BadRequestException("Budget already exists for this category and period");
        }

        Budget budget = Budget.builder()
                .user(currentUser)
                .category(category)
                .monthlyLimit(request.getMonthlyLimit())
                .month(request.getMonth())
                .year(request.getYear())
                .build();

        Budget saved = budgetRepository.save(budget);
        return mapToResponse(saved);
    }

    public List<BudgetResponse> getBudgets(Integer month, Integer year) {
        User currentUser = securityUtils.getCurrentUser();

        List<Budget> budgets;
        if (month != null && year != null) {
            budgets = budgetRepository.findByUserIdAndMonthAndYear(currentUser.getId(), month, year);
        } else {
            LocalDate now = LocalDate.now();
            budgets = budgetRepository.findByUserIdAndMonthAndYear(currentUser.getId(), now.getMonthValue(), now.getYear());
        }

        return budgets.stream().map(this::mapToResponse).collect(Collectors.toList());
    }

    public List<BudgetStatusResponse> getBudgetStatus(int month, int year) {
        User currentUser = securityUtils.getCurrentUser();
        List<Budget> budgets = budgetRepository.findByUserIdAndMonthAndYear(currentUser.getId(), month, year);

        return budgets.stream().map(budget -> {
            List<Transaction> categoryTransactions = transactionRepository
                    .findByUserIdAndCategoryIdOrderByDateDesc(currentUser.getId(), budget.getCategory().getId())
                    .stream()
                    .filter(t -> t.getDate().getMonthValue() == month && t.getDate().getYear() == year)
                    .collect(Collectors.toList());

            BigDecimal amountSpent = categoryTransactions.stream()
                    .filter(t -> t.getType() == Transaction.TransactionType.EXPENSE)
                    .map(Transaction::getAmount)
                    .reduce(BigDecimal.ZERO, BigDecimal::add);

            BigDecimal limit = budget.getMonthlyLimit();
            BigDecimal remaining = limit.subtract(amountSpent);
            double percentageUsed = limit.compareTo(BigDecimal.ZERO) > 0
                    ? amountSpent.divide(limit, 4, RoundingMode.HALF_UP).doubleValue() * 100
                    : 0;

            return BudgetStatusResponse.builder()
                    .budgetId(budget.getId())
                    .categoryId(budget.getCategory().getId())
                    .categoryName(budget.getCategory().getName())
                    .monthlyLimit(limit)
                    .amountSpent(amountSpent)
                    .remaining(remaining)
                    .percentageUsed(percentageUsed)
                    .exceeded(amountSpent.compareTo(limit) > 0)
                    .month(month)
                    .year(year)
                    .build();
        }).collect(Collectors.toList());
    }

    private BudgetResponse mapToResponse(Budget budget) {
        return BudgetResponse.builder()
                .id(budget.getId())
                .categoryId(budget.getCategory().getId())
                .categoryName(budget.getCategory().getName())
                .monthlyLimit(budget.getMonthlyLimit())
                .month(budget.getMonth())
                .year(budget.getYear())
                .build();
    }
}
