package com.andreisichet.basketball_predictor.repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;

import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;

public interface GameRepository extends JpaRepository<Game, Long> {
    /** Genuinely upcoming games, soonest first, with both teams loaded. */
    @EntityGraph(attributePaths = {"homeTeam", "awayTeam"})
    List<Game> findByPlayedFalseAndGameDateGreaterThanEqualOrderByGameDateAsc(
            LocalDate date);

    /** Cached fixtures inside a date window, soonest first. */
    List<Game> findByPlayedFalseAndGameDateBetweenOrderByGameDateAsc(
            LocalDate from, LocalDate to);

    /** Used to reuse an existing game instead of creating a duplicate. */
    Optional<Game> findByHomeTeamAndAwayTeamAndGameDate(Team homeTeam, Team awayTeam, LocalDate gameDate);
}
