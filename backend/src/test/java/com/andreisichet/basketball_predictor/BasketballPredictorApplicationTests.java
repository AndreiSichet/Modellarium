package com.andreisichet.basketball_predictor;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

@SpringBootTest
class BasketballPredictorApplicationTests {

	/**
	 * The test database, refusing the production one.
	 *
	 * BEFORE FLYWAY, WHICH IS THE WHOLE POINT. This context load runs every
	 * pending migration, and a migration cannot be un-applied - so the check
	 * has to happen while the context is being built rather than in a
	 * {@code @BeforeAll}, which runs after it. {@code @DynamicPropertySource}
	 * is evaluated during that build. See {@link TestDatabase}.
	 */
	@DynamicPropertySource
	static void database(DynamicPropertyRegistry registry) {
		registry.add("spring.datasource.url", TestDatabase::url);
	}

	@Test
	void contextLoads() {
	}
}
