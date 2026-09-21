package com.andreisichet.basketball_predictor;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

/** warning, the method simply never fires. */
@SpringBootApplication
@EnableScheduling
public class BasketballPredictorApplication {
	public static void main(String[] args) {
		SpringApplication.run(BasketballPredictorApplication.class, args);
	}
}
