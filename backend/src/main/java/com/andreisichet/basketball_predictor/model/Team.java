package com.andreisichet.basketball_predictor.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/** A team in one of the leagues this app serves. */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Team {
    @Id
    private Long id;

    private String name;

    private String abbreviation;

    /**
     * "NBA" or "WNBA".
     *
     * Load-bearing rather than descriptive: abbreviations COLLIDE across the
     * two leagues - ATL, CHI, DAL, IND, MIN, PHX, TOR and WAS all exist in
     * both - so abbreviation alone no longer identifies a team. Ids do not
     * collide and remain the key; this is what lets a caller ask for one
     * league's teams without getting eight ambiguous rows.
     */
    @Column(nullable = false)
    private String league;
}
