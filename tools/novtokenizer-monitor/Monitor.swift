import SwiftUI
import AppKit

struct Snapshot: Decodable {
    var ok: Bool; var status: String; var observed_at: Double
    var gpus: [GPU]?; var training: Training?; var evaluation: Evaluation?
}
struct GPU: Decodable, Identifiable {
    var index: String; var name: String; var utilization: Double?; var memory_used: Double?; var memory_total: Double?; var power: Double?; var temperature: Double?
    var id: String { index }
}
struct Training: Decodable {
    var chars: Double; var budget: Double; var tokens: Double?; var steps: Double?; var chars_sec: Double?; var tokens_sec: Double?; var elapsed: Double?; var eta: Double?; var complete: Bool; var process_alive: Bool?; var progress_age: Double?; var loss: Double?; var bpb: Double?
}
struct Evaluation: Decodable { var complete: Bool; var finished: Int; var total: Int; var rows: [ResultRow] }
struct ResultRow: Decodable, Identifiable {
    var dataset: String; var seed: Int; var category: String; var works: Int; var j32_bpb: Double; var j48_bpb: Double; var relative_percent: Double
    var id: String { "\(dataset)-\(seed)-\(category)" }
}
@MainActor final class Monitor: ObservableObject {
    @Published var snapshot: Snapshot?
    @Published var status = "接続準備中"
    @Published var busy = false
    @Published var paused = false
    @Published var lastSuccess: Date?
    @Published var now = Date()
    private var timer: Timer?
    init() {
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self = self else { return }
                self.now = Date()
                if !self.paused && Int(self.now.timeIntervalSince1970) % 10 == 0 { self.refresh() }
            }
        }
        refresh()
    }
    var stale: Bool { lastSuccess == nil || now.timeIntervalSince(lastSuccess!) > 30 }
    func refresh() {
        guard !busy else { return }
        busy = true
        let resource = Bundle.main.resourceURL!
        DispatchQueue.global(qos: .utility).async {
            let process = Process(); let pipe = Pipe()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
            process.arguments = [resource.appendingPathComponent("collector.py").path, "--once", "--config", resource.appendingPathComponent("config.json").path]
            process.standardOutput = pipe
            process.standardError = FileHandle.nullDevice
            var data = Data()
            do {
                try process.run()
                data = pipe.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
            } catch { }
            let decoded = try? JSONDecoder().decode(Snapshot.self, from: data)
            Task { @MainActor in
                self.busy = false
                if let value = decoded {
                    self.status = value.status
                    if value.ok { self.snapshot = value; self.lastSuccess = Date(timeIntervalSince1970: value.observed_at) }
                } else { self.status = "収集プロセスの応答を確認してください" }
            }
        }
    }
}
func number(_ value: Double?, _ suffix: String = "", digits: Int = 0) -> String {
    guard let value = value else { return "—" }
    return String(format: "%.*f", digits, value) + suffix
}
func duration(_ value: Double?) -> String {
    guard let value = value, value >= 0 else { return "—" }
    return "\(Int(value) / 3600)時間 \(Int(value) % 3600 / 60)分"
}
struct Card<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View { VStack(alignment: .leading, spacing: 12) { content }.padding(20).frame(maxWidth: .infinity, alignment: .leading).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16)) }
}
struct ContentView: View {
    @StateObject var model = Monitor()
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("NovTokenizer Monitor").font(.largeTitle.bold())
                        Text("Phase 5.5.1 · 実測を10秒ごとに更新 · AI使用なし").foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button(model.paused ? "自動更新を再開" : "更新を一時停止") { model.paused.toggle() }.accessibilityIdentifier("pause")
                    Button("今すぐ更新") { model.refresh() }.disabled(model.busy).accessibilityIdentifier("refresh")
                }
                HStack {
                    Circle().fill(model.stale ? Color.orange : Color.green).frame(width: 8, height: 8)
                    Text(model.paused ? "自動更新停止中" : model.status)
                    if model.busy { ProgressView().controlSize(.small) }
                    Spacer()
                    if let last = model.lastSuccess { Text("最終取得 \(last.formatted(date: .omitted, time: .standard)) · \(Int(max(0, model.now.timeIntervalSince(last))))秒前").foregroundStyle(.secondary) }
                }
                if model.stale { Text("最新データを取得できていません。表示値は最後に取得した実測です。").foregroundStyle(.orange) }
                HStack(alignment: .top, spacing: 16) {
                    ForEach(model.snapshot?.gpus ?? []) { gpu in
                        Card {
                            Text(gpu.name).font(.headline)
                            HStack(alignment: .firstTextBaseline) { Text(number(gpu.utilization, "%")).font(.system(size: 38, weight: .semibold, design: .rounded)); Text("GPU使用率").foregroundStyle(.secondary) }
                            ProgressView(value: gpu.utilization ?? 0, total: 100).tint(.cyan)
                            Text("VRAM  \(number(gpu.memory_used.map { $0 / 1024 }, "", digits: 1)) / \(number(gpu.memory_total.map { $0 / 1024 }, " GiB", digits: 1))")
                            HStack { Text(number(gpu.power, " W", digits: 1)); Spacer(); Text(number(gpu.temperature, " °C")) }.foregroundStyle(.secondary)
                        }
                    }
                }
                Card {
                    HStack { Text("J64 exploratory · seed 1").font(.title2.bold()); Spacer(); Text(trainingState).foregroundStyle(.cyan) }
                    if let t = model.snapshot?.training {
                        HStack { Text(number(min(100, t.chars / max(t.budget, 1) * 100), "%", digits: 2)).font(.system(size: 32, weight: .semibold)); Spacer(); Text("\(number(t.chars)) / \(number(t.budget)) 文字") }
                        ProgressView(value: min(t.chars, t.budget), total: max(t.budget, 1)).tint(.cyan)
                        HStack(spacing: 35) {
                            metric("学習 chars / s", number(t.chars_sec)); metric("学習 tokens / s", number(t.tokens_sec)); metric("経過", duration(t.elapsed)); metric("残り・推定", duration(t.eta))
                        }
                        HStack { Text("step \(number(t.steps)) · tokens \(number(t.tokens)) · loss \(number(t.loss, digits: 4))"); Spacer(); if t.bpb != nil { Text("BPB \(number(t.bpb, digits: 4))") } }.font(.caption).foregroundStyle(.secondary)
                        Text("速度は累積学習時間ベース。ETAは開始からの経過時間と文字数から算出し、評価・保存の時間で変動します。").font(.caption).foregroundStyle(.secondary)
                        if (t.progress_age ?? 0) > 60 && !t.complete { Text("進捗ファイルは\(Int(t.progress_age ?? 0))秒前。評価・checkpoint保存中の可能性があります。").font(.caption).foregroundStyle(.orange) }
                    } else { Text("実測の取得待ち") }
                }
                Card {
                    HStack { Text("漢文・書き下し評価 · P100").font(.title2.bold()); Spacer(); if let e = model.snapshot?.evaluation { Text(e.complete ? "完了 · \(e.finished)/\(e.total)" : "\(e.finished)/\(e.total) 完了").foregroundStyle(.green) } }
                    Text("hardening後の J48 − J32。正の値はBPB悪化です。").foregroundStyle(.secondary)
                    ForEach((model.snapshot?.evaluation?.rows ?? []).filter { $0.dataset == "hardened" }) { row in
                        HStack { Text("\(row.category) · seed \(row.seed)").frame(width: 205, alignment: .leading); Text("\(row.works)作品").frame(width: 90); Text(String(format: "%.4f → %.4f", row.j32_bpb, row.j48_bpb)); Spacer(); Text(String(format: "%+.2f%%", row.relative_percent)).foregroundStyle(row.relative_percent > 0 ? .orange : .green) }.monospacedDigit()
                    }
                }
                Text("読み取り専用 · アプリを閉じると監視も終了 · 学習・評価プロセスは継続します").font(.caption).foregroundStyle(.secondary)
            }.padding(26)
        }.frame(minWidth: 860, minHeight: 740).background(Color(nsColor: .windowBackgroundColor))
    }
    var trainingState: String {
        guard let t = model.snapshot?.training else { return "取得待ち" }
        if t.complete { return "完了" }
        if t.process_alive == false { return "プロセス停止・結果未完了" }
        return t.process_alive == true ? "実行中" : "状態未確認"
    }
    func metric(_ title: String, _ value: String) -> some View { VStack(alignment: .leading, spacing: 4) { Text(title).font(.caption).foregroundStyle(.secondary); Text(value).font(.title3.monospacedDigit()) } }
}
@main struct MonitorApp: App {
    var body: some Scene { WindowGroup { ContentView() }.defaultSize(width: 1040, height: 840) }
}
