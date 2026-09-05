import Foundation
import NeuralApp

final class EngineBox: ObservableObject, @unchecked Sendable {
    static let shared = EngineBox(engine: AppEngine())

    private let lock = NSLock()
    private var engine: AppEngine

    init(engine: AppEngine) {
        self.engine = engine
    }

    func runApp() -> Int32 {
        lock.lock()
        defer { lock.unlock() }
        return engine.runApp(progress_callback)
    }

    func sendRasterData(_ data: UnsafePointer<Float>, _ size: Int) {
        lock.lock()
        defer { lock.unlock() }
        engine.sendRasterData(data, size)
    }
}
