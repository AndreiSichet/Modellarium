package com.andreisichet.basketball_predictor.config;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/** Fills the team table on a database that has none. */
@Component
public class TeamSeeder implements CommandLineRunner {
    private static final Logger log = LoggerFactory.getLogger(TeamSeeder.class);

    private final TeamRepository teamRepository;

    public TeamSeeder(TeamRepository teamRepository) {
        this.teamRepository = teamRepository;
    }

    @Override
    public void run(String... args) {
        long existing = teamRepository.count();
        if (existing > 0) {
            log.debug("Team table already has {} rows; skipping seed.", existing);
            return;
        }

        teamRepository.saveAll(TEAMS);
        log.info("Seeded {} teams", TEAMS.size());
    }

    private static final List<Team> TEAMS = List.of(
            new Team(1610612737L, "Atlanta Hawks", "ATL"),
            new Team(1610612738L, "Boston Celtics", "BOS"),
            new Team(1610612751L, "Brooklyn Nets", "BKN"),
            new Team(1610612766L, "Charlotte Hornets", "CHA"),
            new Team(1610612741L, "Chicago Bulls", "CHI"),
            new Team(1610612739L, "Cleveland Cavaliers", "CLE"),
            new Team(1610612742L, "Dallas Mavericks", "DAL"),
            new Team(1610612743L, "Denver Nuggets", "DEN"),
            new Team(1610612765L, "Detroit Pistons", "DET"),
            new Team(1610612744L, "Golden State Warriors", "GSW"),
            new Team(1610612745L, "Houston Rockets", "HOU"),
            new Team(1610612754L, "Indiana Pacers", "IND"),
            new Team(1610612746L, "LA Clippers", "LAC"),
            new Team(1610612747L, "Los Angeles Lakers", "LAL"),
            new Team(1610612763L, "Memphis Grizzlies", "MEM"),
            new Team(1610612748L, "Miami Heat", "MIA"),
            new Team(1610612749L, "Milwaukee Bucks", "MIL"),
            new Team(1610612750L, "Minnesota Timberwolves", "MIN"),
            new Team(1610612740L, "New Orleans Pelicans", "NOP"),
            new Team(1610612752L, "New York Knicks", "NYK"),
            new Team(1610612760L, "Oklahoma City Thunder", "OKC"),
            new Team(1610612753L, "Orlando Magic", "ORL"),
            new Team(1610612755L, "Philadelphia 76ers", "PHI"),
            new Team(1610612756L, "Phoenix Suns", "PHX"),
            new Team(1610612757L, "Portland Trail Blazers", "POR"),
            new Team(1610612758L, "Sacramento Kings", "SAC"),
            new Team(1610612759L, "San Antonio Spurs", "SAS"),
            new Team(1610612761L, "Toronto Raptors", "TOR"),
            new Team(1610612762L, "Utah Jazz", "UTA"),
            new Team(1610612764L, "Washington Wizards", "WAS"));
}
