package com.andreisichet.basketball_predictor.service;

import java.time.LocalDate;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ResponseStatusException;

import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/** Team resolution and find-or-create for a Game. */
@Component
public class GameLookup {
    private final TeamRepository teamRepository;
    private final GameRepository gameRepository;

    public GameLookup(TeamRepository teamRepository, GameRepository gameRepository) {
        this.teamRepository = teamRepository;
        this.gameRepository = gameRepository;
    }

    /** The team, or a 400 naming the id that was not recognised. */
    public Team requireTeam(Long teamId) {
        if (teamId == null) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Team id is required.");
        }
        return teamRepository.findById(teamId)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.BAD_REQUEST, "Unknown team id: " + teamId));
    }

    /** The existing Game for this matchup and date, or a new one. */
    public Game findOrCreateGame(Team homeTeam, Team awayTeam, LocalDate gameDate) {
        return gameRepository.findByHomeTeamAndAwayTeamAndGameDate(homeTeam, awayTeam, gameDate)
                .orElseGet(() -> {
                    Game game = new Game();
                    game.setHomeTeam(homeTeam);
                    game.setAwayTeam(awayTeam);
                    game.setGameDate(gameDate);
                    game.setPlayed(false);
                    return gameRepository.save(game);
                });
    }
}
