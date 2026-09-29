import XCTest
@testable import EmojiModel

/// The browser implementation's answers (app/fixtures/parity.json, written by
/// app/sync_model.sh) are the reference this port must reproduce.
final class ParityTests: XCTestCase {
    private struct Expected: Decodable {
        let tokens: [Int32]
        let emoji: String
        let maxProb: Double
        let topIdx: Int
    }

    static let model: EmojiModel = try! EmojiModel.load()

    /// PARITY_FIXTURE points at another check.mjs output, e.g. a larger fuzz set.
    private func fixture() throws -> [String: Expected] {
        if let path = ProcessInfo.processInfo.environment["PARITY_FIXTURE"] {
            return try JSONDecoder().decode([String: Expected].self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        }
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // EmojiModelTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // EmojiModel
            .deletingLastPathComponent()   // ios
            .deletingLastPathComponent()   // app
            .appendingPathComponent("fixtures/parity.json")
        return try JSONDecoder().decode([String: Expected].self, from: Data(contentsOf: url))
    }

    func testMatchesJavaScript() throws {
        let cases = try fixture()
        XCTAssertGreaterThan(cases.count, 0)
        let model = Self.model
        var worst = 0.0
        for (phrase, want) in cases.sorted(by: { $0.key < $1.key }) {
            let label = phrase.debugDescription
            XCTAssertEqual(model.tokenize(phrase), want.tokens, "tokens for \(label)")

            let probs = model.probabilities(phrase)
            var top = 0
            for i in 1..<probs.count where probs[i] > probs[top] { top = i }
            XCTAssertEqual(top, want.topIdx, "topIdx for \(label)")
            XCTAssertEqual(Double(probs[top]), want.maxProb, accuracy: 2e-4, "maxProb for \(label)")
            worst = max(worst, abs(Double(probs[top]) - want.maxProb))

            let emoji = model.predict(phrase).map(\.emoji).joined()
            XCTAssertEqual(Array(emoji.unicodeScalars), Array(want.emoji.unicodeScalars), "emoji for \(label)")
        }
        print("parity: \(cases.count) phrases, worst maxProb deviation \(worst)")
    }

    func testFinalSigmaLowercasesLikeJavaScript() {
        XCTAssertEqual(jsLowercased("ΟΔΟΣ ΣΑ"), "οδος σα")
        XCTAssertEqual(jsLowercased("Σ"), "σ")
    }

    func testRepeatsCollapseOnCodeUnits() {
        XCTAssertEqual(cleanText("noooooo"), "nooo")
        // JS `.` without the u flag sees a surrogate pair as two different units.
        XCTAssertEqual(cleanText("😂😂😂😂😂"), "😂😂😂😂😂")
    }

    func testPredictionIsFast() {
        let model = Self.model
        let phrases = ["Pizza tonight?", "We won the game last night!", "Going to Spain next week",
                       "Congrats on the new job!!", "It's raining all day here"]
        for p in phrases { _ = model.predict(p) }
        let rounds = 10
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<rounds { for p in phrases { _ = model.predict(p) } }
        let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(rounds * phrases.count)
        print("average predict: \(String(format: "%.2f", ms)) ms")
        XCTAssertLessThan(ms, 50)
    }
}
