package com.andreisichet.basketball_predictor.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import com.andreisichet.basketball_predictor.model.Team;

public interface TeamRepository extends JpaRepository<Team, Long> {
    List<Team> findByLeagueOrderByNameAsc(String league);

    long countByLeague(String league);
}
