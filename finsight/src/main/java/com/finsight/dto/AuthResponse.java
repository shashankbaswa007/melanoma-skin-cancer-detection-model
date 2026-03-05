package com.finsight.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AuthResponse {
    private String token;
    private String type = "Bearer";
    private String email;
    private String name;

    public AuthResponse(String token, String email, String name) {
        this.token = token;
        this.type = "Bearer";
        this.email = email;
        this.name = name;
    }
}
