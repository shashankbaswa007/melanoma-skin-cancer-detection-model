package com.finsight.dto;

import com.finsight.model.Category;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class CategoryRequest {
    @NotBlank
    private String name;

    @NotNull
    private Category.TransactionType type;
}
