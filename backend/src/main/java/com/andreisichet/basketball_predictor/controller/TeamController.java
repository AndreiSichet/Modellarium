package com.andreisichet.basketball_predictor.controller;

import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.andreisichet.basketball_predictor.dto.TeamDto;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/** Calls the repository directly. */
@RestController
@RequestMapping("/api/teams")
public class TeamController {
    private final TeamRepository teamRepository;

    public TeamController(TeamRepository teamRepository) {
        this.teamRepository = teamRepository;
    }

    /**
     * One league's teams, defaulting to the NBA.
     *
     * THE DEFAULT IS A LEAGUE, NOT ALL LEAGUES, and that is the whole point.
     * findAll() would now return 45 rows including eight duplicated
     * abbreviations (ATL, CHI, DAL, IND, MIN, PHX, TOR, WAS belong to a team
     * in each league), so an existing caller populating an NBA team picker
     * would start showing two Atlantas. Keeping the default at NBA leaves
     * every current response byte-identical and makes the second league
     * something a caller opts into.
     */
    @GetMapping
    public List<TeamDto> byLeague(@RequestParam(defaultValue = "NBA") String league) {
        return teamRepository.findByLeagueOrderByNameAsc(league).stream()
                .map(TeamDto::from)
                .toList();
    }
}
