package com.finsight.repository;

import com.finsight.model.Category;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface CategoryRepository extends JpaRepository<Category, Long> {
    List<Category> findByUserId(Long userId);
    List<Category> findByUserIdOrUserIsNull(Long userId);
    Optional<Category> findByIdAndUserId(Long id, Long userId);
}
