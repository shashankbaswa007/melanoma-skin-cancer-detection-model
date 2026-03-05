package com.finsight.service;

import com.finsight.dto.TransactionRequest;
import com.finsight.dto.TransactionResponse;
import com.finsight.exception.BadRequestException;
import com.finsight.exception.ResourceNotFoundException;
import com.finsight.exception.UnauthorizedException;
import com.finsight.model.Category;
import com.finsight.model.Transaction;
import com.finsight.model.User;
import com.finsight.repository.CategoryRepository;
import com.finsight.repository.TransactionRepository;
import com.finsight.util.SecurityUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class TransactionService {

    private final TransactionRepository transactionRepository;
    private final CategoryRepository categoryRepository;
    private final SecurityUtils securityUtils;

    public TransactionResponse createTransaction(TransactionRequest request) {
        User currentUser = securityUtils.getCurrentUser();

        Category category = null;
        if (request.getCategoryId() != null) {
            category = categoryRepository.findById(request.getCategoryId())
                    .orElseThrow(() -> new ResourceNotFoundException("Category not found with id: " + request.getCategoryId()));
            // Ensure the category is either a default (no owner) or belongs to the current user
            if (category.getUser() != null && !category.getUser().getId().equals(currentUser.getId())) {
                throw new UnauthorizedException("You are not authorized to use this category");
            }
        }

        Transaction transaction = Transaction.builder()
                .user(currentUser)
                .amount(request.getAmount())
                .type(request.getType())
                .category(category)
                .description(request.getDescription())
                .date(request.getDate())
                .build();

        Transaction saved = transactionRepository.save(transaction);
        return mapToResponse(saved);
    }

    public TransactionResponse updateTransaction(Long id, TransactionRequest request) {
        User currentUser = securityUtils.getCurrentUser();

        Transaction transaction = transactionRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Transaction not found with id: " + id));

        if (!transaction.getUser().getId().equals(currentUser.getId())) {
            throw new UnauthorizedException("You are not authorized to update this transaction");
        }

        Category category = null;
        if (request.getCategoryId() != null) {
            category = categoryRepository.findById(request.getCategoryId())
                    .orElseThrow(() -> new ResourceNotFoundException("Category not found with id: " + request.getCategoryId()));
            // Ensure the category is either a default (no owner) or belongs to the current user
            if (category.getUser() != null && !category.getUser().getId().equals(currentUser.getId())) {
                throw new UnauthorizedException("You are not authorized to use this category");
            }
        }

        transaction.setAmount(request.getAmount());
        transaction.setType(request.getType());
        transaction.setCategory(category);
        transaction.setDescription(request.getDescription());
        transaction.setDate(request.getDate());

        Transaction updated = transactionRepository.save(transaction);
        return mapToResponse(updated);
    }

    public void deleteTransaction(Long id) {
        User currentUser = securityUtils.getCurrentUser();

        Transaction transaction = transactionRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Transaction not found with id: " + id));

        if (!transaction.getUser().getId().equals(currentUser.getId())) {
            throw new UnauthorizedException("You are not authorized to delete this transaction");
        }

        transactionRepository.delete(transaction);
    }

    public Page<TransactionResponse> getTransactions(Long categoryId, Pageable pageable) {
        User currentUser = securityUtils.getCurrentUser();

        Page<Transaction> transactions;
        if (categoryId != null) {
            transactions = transactionRepository.findByUserIdAndCategoryIdOrderByDateDesc(currentUser.getId(), categoryId, pageable);
        } else {
            transactions = transactionRepository.findByUserIdOrderByDateDesc(currentUser.getId(), pageable);
        }

        return transactions.map(this::mapToResponse);
    }

    public List<TransactionResponse> getMonthlyTransactions(int month, int year) {
        User currentUser = securityUtils.getCurrentUser();
        List<Transaction> transactions = transactionRepository.findMonthlyTransactions(currentUser.getId(), year, month);
        return transactions.stream().map(this::mapToResponse).collect(Collectors.toList());
    }

    private TransactionResponse mapToResponse(Transaction transaction) {
        return TransactionResponse.builder()
                .id(transaction.getId())
                .amount(transaction.getAmount())
                .type(transaction.getType().name())
                .categoryId(transaction.getCategory() != null ? transaction.getCategory().getId() : null)
                .categoryName(transaction.getCategory() != null ? transaction.getCategory().getName() : null)
                .description(transaction.getDescription())
                .date(transaction.getDate())
                .createdAt(transaction.getCreatedAt())
                .build();
    }
}
