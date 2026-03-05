package com.finsight.service;

import com.finsight.dto.CategoryRequest;
import com.finsight.dto.CategoryResponse;
import com.finsight.model.Category;
import com.finsight.model.User;
import com.finsight.repository.CategoryRepository;
import com.finsight.util.SecurityUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class CategoryService {

    private final CategoryRepository categoryRepository;
    private final SecurityUtils securityUtils;

    public CategoryResponse createCategory(CategoryRequest request) {
        User currentUser = securityUtils.getCurrentUser();

        Category category = Category.builder()
                .name(request.getName())
                .type(request.getType())
                .user(currentUser)
                .build();

        Category saved = categoryRepository.save(category);
        return mapToResponse(saved);
    }

    public List<CategoryResponse> getCategories() {
        User currentUser = securityUtils.getCurrentUser();
        List<Category> categories = categoryRepository.findByUserIdOrUserIsNull(currentUser.getId());
        return categories.stream().map(this::mapToResponse).collect(Collectors.toList());
    }

    private CategoryResponse mapToResponse(Category category) {
        return CategoryResponse.builder()
                .id(category.getId())
                .name(category.getName())
                .type(category.getType().name())
                .isDefault(category.getUser() == null)
                .build();
    }
}
