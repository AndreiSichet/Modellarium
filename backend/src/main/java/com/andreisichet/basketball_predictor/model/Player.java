package com.andreisichet.basketball_predictor.model;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/** An NBA player. */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Player {
    @Id
    private Long id;

    private String name;
}
