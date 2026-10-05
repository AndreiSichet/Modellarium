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
        seed(NBA, NBA_TEAMS, false);
        seed(WNBA, WNBA_TEAMS, false);
        seed(GLEAGUE, GLEAGUE_TEAMS, true);
    }

    private void seed(String league, List<Team> teams, boolean reconcileNames) {
        long existing = teamRepository.countByLeague(league);
        if (existing == 0) {
            teamRepository.saveAll(teams);
            log.info("Seeded {} {} teams", teams.size(), league);
            return;
        }
        if (!reconcileNames) {
            log.debug("{} already has {} teams; skipping seed.", league, existing);
            return;
        }
        reconcile(league, teams);
    }

    /**
     * Bring existing rows' names and abbreviations up to date.
     *
     * ON FOR THE G LEAGUE AND OFF FOR THE OTHER TWO, decided on measurement
     * rather than on symmetry. Phase 1's identity table records that 19 of 40
     * G League franchises changed name or abbreviation at least once under a
     * stable id - Asheville Altitude became Tulsa 66ers and then Oklahoma
     * City Blue, all on 1612709889 - against 3 of 15 for the WNBA and 0 of 30
     * for the NBA.
     *
     * An insert-only seeder would therefore keep a franchise's OLD name for
     * as long as the row exists, and a stale name on a live fixture is a
     * visible error. Reconciling costs 31 reads and usually zero writes.
     *
     * Off for the NBA and WNBA deliberately: their names are stable, and the
     * regression gate forbids this change touching their serving at all. A
     * per-league flag rather than a special case, so a fourth league has to
     * state its own answer.
     */
    private void reconcile(String league, List<Team> teams) {
        int updated = 0;
        for (Team wanted : teams) {
            Team current = teamRepository.findById(wanted.getId()).orElse(null);
            if (current == null) {
                teamRepository.save(wanted);
                updated++;
                continue;
            }
            boolean nameChanged = !wanted.getName().equals(current.getName());
            boolean abbrChanged =
                    !wanted.getAbbreviation().equals(current.getAbbreviation());
            if (nameChanged || abbrChanged) {
                log.info("{} team {} renamed: {} ({}) -> {} ({})", league,
                        current.getId(), current.getName(),
                        current.getAbbreviation(), wanted.getName(),
                        wanted.getAbbreviation());
                current.setName(wanted.getName());
                current.setAbbreviation(wanted.getAbbreviation());
                teamRepository.save(current);
                updated++;
            }
        }
        if (updated > 0) {
            log.info("Reconciled {} {} team row(s)", updated, league);
        } else {
            log.debug("{} teams already current; nothing reconciled", league);
        }
    }

    /**
     * All three leagues' ids must be pairwise disjoint, because Team.id is
     * the primary key and every league shares the table.
     *
     * Checked rather than assumed, and all three pairs rather than the one
     * that used to exist. They are disjoint today by a wide margin - NBA
     * 1610612737-1610612766, WNBA 1611661313-1611661332, G League
     * 1612709889-1612709934 - but the ids come from nba_api and nothing in
     * this codebase controls them, so a future expansion team landing on an
     * existing id would otherwise surface as one league's row silently
     * overwriting another's.
     */
    private void assertNoIdCollision() {
        checkDisjoint(NBA, NBA_TEAMS, WNBA, WNBA_TEAMS);
        checkDisjoint(NBA, NBA_TEAMS, GLEAGUE, GLEAGUE_TEAMS);
        checkDisjoint(WNBA, WNBA_TEAMS, GLEAGUE, GLEAGUE_TEAMS);
    }

    private void checkDisjoint(String leftName, List<Team> left,
            String rightName, List<Team> right) {
        Set<Long> leftIds = left.stream().map(Team::getId)
                .collect(Collectors.toSet());
        Set<Long> overlap = right.stream()
                .map(Team::getId)
                .filter(leftIds::contains)
                .collect(Collectors.toSet());
        if (!overlap.isEmpty()) {
            throw new IllegalStateException(
                    leftName + " and " + rightName + " team ids collide on "
                            + overlap
                            + ". Team.id is the primary key and all leagues share"
                            + " the table, so one league's row would overwrite"
                            + " another's.");
        }
    }

    private static final String NBA = "NBA";
    private static final String WNBA = "WNBA";
    private static final String GLEAGUE = "GLEAGUE";

    private static Team nba(long id, String name, String abbreviation) {
        return new Team(id, name, abbreviation, NBA);
    }

    private static Team wnba(long id, String name, String abbreviation) {
        return new Team(id, name, abbreviation, WNBA);
    }

    private static Team gleague(long id, String name, String abbreviation) {
        return new Team(id, name, abbreviation, GLEAGUE);
    }

    /**
     * The 31 franchises active in 2025-26, generated from phase 1's identity
     * table rather than typed from memory.
     *
     * Historical ids that no longer play are deliberately absent: nothing
     * references them, and a Game row can only be created for a team the
     * schedule sync resolved. If one is ever needed, it is a row here.
     */
    private static final List<Team> GLEAGUE_TEAMS = List.of(
            gleague(1612709890L, "Austin Spurs", "AUS"),
            gleague(1612709913L, "Birmingham Squadron", "BHM"),
            gleague(1612709928L, "Capital City Go-Go", "CCG"),
            gleague(1612709893L, "Cleveland Charge", "CLC"),
            gleague(1612709929L, "College Park Skyhawks", "CPS"),
            gleague(1612709909L, "Delaware Blue Coats", "DEL"),
            gleague(1612709922L, "Greensboro Swarm", "GBO"),
            gleague(1612709917L, "Grand Rapids Gold", "GRG"),
            gleague(1612709911L, "Iowa Wolves", "IWA"),
            gleague(1612709921L, "Long Island Nets", "LIN"),
            gleague(1612709932L, "Motor City Cruise", "MCC"),
            gleague(1612709926L, "Memphis Hustle", "MHU"),
            gleague(1612709915L, "Maine Celtics", "MNE"),
            gleague(1612709931L, "Mexico City Capitanes", "MXC"),
            gleague(1612709910L, "Noblesville Boom", "NOB"),
            gleague(1612709889L, "Oklahoma City Blue", "OKL"),
            gleague(1612709925L, "Osceola Magic", "OSC"),
            gleague(1612709920L, "Raptors 905", "RAP"),
            gleague(1612709933L, "Rip City Remix", "RCR"),
            gleague(1612709908L, "Rio Grande Valley Vipers", "RGV"),
            gleague(1612709905L, "South Bay Lakers", "SBL"),
            gleague(1612709902L, "Santa Cruz Warriors", "SCW"),
            gleague(1612709924L, "San Diego Clippers", "SDC"),
            gleague(1612709903L, "Salt Lake City Stars", "SLC"),
            gleague(1612709914L, "Stockton Kings", "STO"),
            gleague(1612709904L, "Sioux Falls Skyforce", "SXF"),
            gleague(1612709918L, "Texas Legends", "TEX"),
            gleague(1612709934L, "Valley Suns", "VAL"),
            gleague(1612709923L, "Windy City Bulls", "WCB"),
            gleague(1612709919L, "Westchester Knicks", "WES"),
            gleague(1612709927L, "Wisconsin Herd", "WIS"));

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
