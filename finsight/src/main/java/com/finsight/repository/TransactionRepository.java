package com.finsight.repository;

import com.finsight.model.Transaction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;

@Repository
public interface TransactionRepository extends JpaRepository<Transaction, Long> {

    List<Transaction> findByUserIdOrderByDateDesc(Long userId);

    List<Transaction> findByUserIdAndCategoryIdOrderByDateDesc(Long userId, Long categoryId);

    @Query("SELECT t FROM Transaction t WHERE t.user.id = :userId AND YEAR(t.date) = :year AND MONTH(t.date) = :month ORDER BY t.date DESC")
    List<Transaction> findMonthlyTransactions(@Param("userId") Long userId, @Param("year") int year, @Param("month") int month);

    @Query("SELECT t FROM Transaction t WHERE t.user.id = :userId AND t.date BETWEEN :startDate AND :endDate")
    List<Transaction> findByUserIdAndDateBetween(@Param("userId") Long userId, @Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);

    @Query("SELECT SUM(t.amount) FROM Transaction t WHERE t.user.id = :userId AND t.type = :type AND YEAR(t.date) = :year AND MONTH(t.date) = :month")
    BigDecimal sumByUserIdAndTypeAndMonth(@Param("userId") Long userId, @Param("type") Transaction.TransactionType type, @Param("year") int year, @Param("month") int month);

    @Query("SELECT t.category.name, SUM(t.amount) FROM Transaction t WHERE t.user.id = :userId AND t.type = 'EXPENSE' AND YEAR(t.date) = :year AND MONTH(t.date) = :month GROUP BY t.category.name ORDER BY SUM(t.amount) DESC")
    List<Object[]> findTopSpendingCategories(@Param("userId") Long userId, @Param("year") int year, @Param("month") int month);

    @Query("SELECT AVG(t.amount) FROM Transaction t WHERE t.user.id = :userId AND t.type = 'EXPENSE'")
    BigDecimal findAverageExpenseAmount(@Param("userId") Long userId);

    @Query("SELECT STDDEV(t.amount) FROM Transaction t WHERE t.user.id = :userId AND t.type = 'EXPENSE'")
    BigDecimal findStdDevExpenseAmount(@Param("userId") Long userId);

    @Query("SELECT t FROM Transaction t WHERE t.user.id = :userId AND t.type = 'EXPENSE' AND YEAR(t.date) = :year AND MONTH(t.date) = :month ORDER BY t.date DESC")
    List<Transaction> findExpensesByUserIdAndMonth(@Param("userId") Long userId, @Param("year") int year, @Param("month") int month);

    @Query("SELECT t FROM Transaction t WHERE t.user.id = :userId AND t.type = 'EXPENSE' ORDER BY t.date DESC")
    List<Transaction> findExpensesByUserId(@Param("userId") Long userId);
}
