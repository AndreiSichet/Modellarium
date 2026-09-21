package com.andreisichet.basketball_predictor.dto;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

/** Loads a raw inference-service response from src/test/resources/fixtures/. */
final class Fixture {
    private Fixture() {
    }

    static String read(String name) {
        String path = "/fixtures/" + name;
        try (InputStream stream = Fixture.class.getResourceAsStream(path)) {
            if (stream == null) {
                throw new IllegalStateException("no fixture on the classpath at " + path);
            }
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException error) {
            throw new UncheckedIOException("could not read " + path, error);
        }
    }
}
