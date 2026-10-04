package com.andreisichet.basketball_predictor.config;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

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

    /**
     * Seeds PER LEAGUE, not on the table being empty.
     *
     * The original guard was teamRepository.count() == 0, which was right
     * while there was one league. It is now actively wrong: a database seeded
     * before WNBA support has 30 rows, so a total-count guard skips, the 15
     * WNBA teams never arrive, and every WNBA prediction fails with "unknown
     * team id" on a database that looks correctly seeded. Counting per league
     * handles the fresh database and the already-seeded one with the same
     * statement.
     */
    @Override
    public void run(String... args) {
        assertNoIdCollision();
        seed(NBA, NBA_TEAMS);
        seed(WNBA, WNBA_TEAMS);
    }

    private void seed(String league, List<Team> teams) {
        long existing = teamRepository.countByLeague(league);
        if (existing > 0) {
            log.debug("{} already has {} teams; skipping seed.", league, existing);
            return;
        }
        teamRepository.saveAll(teams);
        log.info("Seeded {} {} teams", teams.size(), league);
    }

    /**
     * The two leagues' ids must be disjoint, because Team.id is the primary
     * key and both leagues share the table.
     *
     * Checked rather than assumed. They are disjoint today by a wide margin -
     * NBA 1610612737-1610612766 against WNBA 1611661313-1611661332 - but the
     * ids come from nba_api and nothing in this codebase controls them, so a
     * future expansion team landing on an existing id would otherwise surface
     * as one league's row silently overwriting the other's.
     */
    private void assertNoIdCollision() {
        Set<Long> nbaIds = NBA_TEAMS.stream().map(Team::getId).collect(Collectors.toSet());
        Set<Long> overlap = WNBA_TEAMS.stream()
                .map(Team::getId)
                .filter(nbaIds::contains)
                .collect(Collectors.toSet());
        if (!overlap.isEmpty()) {
            throw new IllegalStateException(
                    "NBA and WNBA team ids collide on " + overlap
                            + ". Team.id is the primary key and both leagues share"
                            + " the table, so one league's row would overwrite the"
                            + " other's.");
        }
    }

    private static final String NBA = "NBA";
    private static final String WNBA = "WNBA";

    private static Team nba(long id, String name, String abbreviation) {
        return new Team(id, name, abbreviation, NBA);
    }

    private static Team wnba(long id, String name, String abbreviation) {
        return new Team(id, name, abbreviation, WNBA);
    }

    private static final List<Team> NBA_TEAMS = List.of(
            nba(1610612737L, "Atlanta Hawks", "ATL"),
            nba(1610612738L, "Boston Celtics", "BOS"),
            nba(1610612751L, "Brooklyn Nets", "BKN"),
            nba(1610612766L, "Charlotte Hornets", "CHA"),
            nba(1610612741L, "Chicago Bulls", "CHI"),
            nba(1610612739L, "Cleveland Cavaliers", "CLE"),
            nba(1610612742L, "Dallas Mavericks", "DAL"),
            nba(1610612743L, "Denver Nuggets", "DEN"),
            nba(1610612765L, "Detroit Pistons", "DET"),
            nba(1610612744L, "Golden State Warriors", "GSW"),
            nba(1610612745L, "Houston Rockets", "HOU"),
            nba(1610612754L, "Indiana Pacers", "IND"),
            nba(1610612746L, "LA Clippers", "LAC"),
            nba(1610612747L, "Los Angeles Lakers", "LAL"),
            nba(1610612763L, "Memphis Grizzlies", "MEM"),
            nba(1610612748L, "Miami Heat", "MIA"),
            nba(1610612749L, "Milwaukee Bucks", "MIL"),
            nba(1610612750L, "Minnesota Timberwolves", "MIN"),
            nba(1610612740L, "New Orleans Pelicans", "NOP"),
            nba(1610612752L, "New York Knicks", "NYK"),
            nba(1610612760L, "Oklahoma City Thunder", "OKC"),
            nba(1610612753L, "Orlando Magic", "ORL"),
            nba(1610612755L, "Philadelphia 76ers", "PHI"),
            nba(1610612756L, "Phoenix Suns", "PHX"),
            nba(1610612757L, "Portland Trail Blazers", "POR"),
            nba(1610612758L, "Sacramento Kings", "SAC"),
            nba(1610612759L, "San Antonio Spurs", "SAS"),
            nba(1610612761L, "Toronto Raptors", "TOR"),
            nba(1610612762L, "Utah Jazz", "UTA"),
            nba(1610612764L, "Washington Wizards", "WAS"));

    /**
     * The 15 WNBA teams, hardcoded for the same reasons the NBA list is:
     * static, tiny, and the pipeline CSVs are gitignored and deliberately
     * absent from the backend image. Verified against the distinct
     * TEAM_ID/TEAM_NAME/TEAM_ABBREVIATION triples in
     * data-pipeline/data/wnba/raw/*.csv.
     *
     * Eight of these abbreviations also belong to an NBA team, which is why
     * Team carries a league column and why GET /api/teams filters by one.
     */
    private static final List<Team> WNBA_TEAMS = List.of(
            wnba(1611661330L, "Atlanta Dream", "ATL"),
            wnba(1611661329L, "Chicago Sky", "CHI"),
            wnba(1611661323L, "Connecticut Sun", "CON"),
            wnba(1611661321L, "Dallas Wings", "DAL"),
            wnba(1611661331L, "Golden State Valkyries", "GSV"),
            wnba(1611661325L, "Indiana Fever", "IND"),
            wnba(1611661319L, "Las Vegas Aces", "LVA"),
            wnba(1611661320L, "Los Angeles Sparks", "LAS"),
            wnba(1611661324L, "Minnesota Lynx", "MIN"),
            wnba(1611661313L, "New York Liberty", "NYL"),
            wnba(1611661317L, "Phoenix Mercury", "PHX"),
            wnba(1611661327L, "Portland Fire", "PDX"),
            wnba(1611661328L, "Seattle Storm", "SEA"),
            wnba(1611661332L, "Toronto Tempo", "TOR"),
            wnba(1611661322L, "Washington Mystics", "WAS"));
}
