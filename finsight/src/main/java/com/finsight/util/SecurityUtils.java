package com.finsight.util;

import com.finsight.exception.UnauthorizedException;
import com.finsight.model.User;
import com.finsight.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class SecurityUtils {

    private final UserRepository userRepository;

    public User getCurrentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || !auth.isAuthenticated() || "anonymousUser".equals(auth.getPrincipal())) {
            throw new UnauthorizedException("User not authenticated");
        }
        String email = auth.getName();
        return userRepository.findByEmail(email)
                .orElseThrow(() -> new UnauthorizedException("User not authenticated"));
    }

    public Long getCurrentUserId() {
        return getCurrentUser().getId();
    }
}
