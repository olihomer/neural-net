import Foundation
import NeuralApp

enum ActivationChoice: Int32, CaseIterable, Identifiable, Sendable {
    case relu = 0
    case sigmoid = 1
    case softmax = 2

    var id: Int32 { rawValue }

    var title: String {
        switch self {
        case .relu:
            return "ReLU"
        case .sigmoid:
            return "Sigmoid"
        case .softmax:
            return "Softmax"
        }
    }
}

struct TrainingSettings: Sendable {
    var hiddenLayerSize: Int = 128
    var epochs: Int = 500
    var trainingExamples: Int = 100
    var batchSize: Int = 50
    var learningRate: Double = 0.5
    var hiddenActivation: ActivationChoice = .relu
    var outputActivation: ActivationChoice = .softmax
}

final class EngineBox: ObservableObject, @unchecked Sendable {
    static let shared = EngineBox(engine: AppEngine())

    private let lock = NSLock()
    private var engine: AppEngine

    init(engine: AppEngine) {
        self.engine = engine
    }

    func runApp(settings: TrainingSettings) -> Int32 {
        lock.lock()
        defer { lock.unlock() }
        return engine.runApp(
            progress_callback,
            CInt(settings.hiddenLayerSize),
            CInt(settings.epochs),
            CInt(settings.trainingExamples),
            CInt(settings.batchSize),
            settings.learningRate,
            CInt(settings.hiddenActivation.rawValue),
            CInt(settings.outputActivation.rawValue)
        )
    }

    func sendRasterData(_ data: UnsafePointer<Float>, _ size: Int) -> (Int, Float) {
        lock.lock()
        defer { lock.unlock() }
        let result = engine.sendRasterData(data, size)
        return (Int(result.first), Float(result.second))
    }

    func saveNetwork(to path: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return path.withCString { engine.saveNetwork($0) }
    }

    func loadNetwork(from path: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return path.withCString { engine.loadNetwork($0) }
    }
}
