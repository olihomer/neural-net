import Foundation
import NeuralApp

enum ModelChoice: Int32, CaseIterable, Identifiable, Sendable {
    case mlp = 0
    case cnn = 1

    var id: Int32 { rawValue }

    var title: String {
        switch self {
        case .mlp:
            return "MLP"
        case .cnn:
            return "CNN"
        }
    }
}

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
    var model: ModelChoice = .cnn
    var hiddenLayerSize: Int = 128
    var epochs: Int = 50
    var trainingExamples: Int = 1000
    var batchSize: Int = 50
    var learningRate: Double = 0.05
    var hiddenActivation: ActivationChoice = .relu
    var outputActivation: ActivationChoice = .softmax
    var cnnConv1Channels: Int = 8
    var cnnConv2Channels: Int = 16
    var cnnClassifierHiddenLayerSize: Int = 128
}

struct KernelVisualizationSnapshot: Sendable {
    let layers: [KernelLayerSnapshot]
    let maxMagnitude: Float
}

struct KernelLayerSnapshot: Identifiable, Sendable {
    let id: Int
    let name: String
    let outputChannels: Int
    let inputChannels: Int
    let kernelHeight: Int
    let kernelWidth: Int
    let kernels: [KernelSnapshot]
}

struct KernelSnapshot: Identifiable, Sendable {
    let id: String
    let layer: Int
    let outputChannel: Int
    let inputChannel: Int
    let width: Int
    let height: Int
    let values: [Float]
}

struct ActivationVisualizationSnapshot: Sendable {
    let sections: [ActivationSectionSnapshot]
    let maxMagnitude: Float
}

struct ActivationSectionSnapshot: Identifiable, Sendable {
    let id: String
    let name: String
    let width: Int
    let height: Int
    let maps: [ActivationMapSnapshot]
}

struct ActivationMapSnapshot: Identifiable, Sendable {
    let id: String
    let channel: Int
    let width: Int
    let height: Int
    let values: [Float]
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
            CInt(settings.model.rawValue),
            CInt(settings.hiddenLayerSize),
            CInt(settings.epochs),
            CInt(settings.trainingExamples),
            CInt(settings.batchSize),
            settings.learningRate,
            CInt(settings.hiddenActivation.rawValue),
            CInt(settings.outputActivation.rawValue),
            CInt(settings.cnnConv1Channels),
            CInt(settings.cnnConv2Channels),
            CInt(settings.cnnClassifierHiddenLayerSize)
        )
    }

    func cnnKernelSnapshot() -> KernelVisualizationSnapshot {
        lock.lock()
        defer { lock.unlock() }

        var layers: [KernelLayerSnapshot] = []
        var maxMagnitude: Float = 0.0

        for layer in [1, 2] {
            let outputChannels = Int(engine.cnnKernelOutputChannels(CInt(layer)))
            let inputChannels = Int(engine.cnnKernelInputChannels(CInt(layer)))
            let height = Int(engine.cnnKernelHeight(CInt(layer)))
            let width = Int(engine.cnnKernelWidth(CInt(layer)))
            var kernels: [KernelSnapshot] = []

            guard outputChannels > 0, inputChannels > 0, height > 0, width > 0 else {
                continue
            }

            for outputChannel in 0..<outputChannels {
                for inputChannel in 0..<inputChannels {
                    var values: [Float] = []
                    values.reserveCapacity(width * height)

                    for y in 0..<height {
                        for x in 0..<width {
                            let value = Float(engine.cnnKernelValue(
                                CInt(layer),
                                CInt(outputChannel),
                                CInt(inputChannel),
                                CInt(y),
                                CInt(x)
                            ))
                            maxMagnitude = max(maxMagnitude, abs(value))
                            values.append(value)
                        }
                    }

                    kernels.append(KernelSnapshot(
                        id: "\(layer)-\(outputChannel)-\(inputChannel)",
                        layer: layer,
                        outputChannel: outputChannel,
                        inputChannel: inputChannel,
                        width: width,
                        height: height,
                        values: values
                    ))
                }
            }

            layers.append(KernelLayerSnapshot(
                id: layer,
                name: layer == 1 ? "Conv1" : "Conv2",
                outputChannels: outputChannels,
                inputChannels: inputChannels,
                kernelHeight: height,
                kernelWidth: width,
                kernels: kernels
            ))
        }

        return KernelVisualizationSnapshot(layers: layers, maxMagnitude: max(maxMagnitude, 0.000001))
    }

    func cnnActivationSnapshot() -> ActivationVisualizationSnapshot {
        lock.lock()
        defer { lock.unlock() }

        var sections: [ActivationSectionSnapshot] = []
        var maxMagnitude: Float = 0.0

        if let inputSection = buildInputSection(layer: 1, name: "Conv1 Input") {
            maxMagnitude = max(maxMagnitude, inputSection.maps.flatMap(\.values).map(abs).max() ?? 0.0)
            sections.append(inputSection)
        }

        for layer in [1, 2] {
            if let activationSection = buildActivationSection(layer: layer, name: layer == 1 ? "Conv1 Feature Maps" : "Conv2 Feature Maps") {
                maxMagnitude = max(maxMagnitude, activationSection.maps.flatMap(\.values).map(abs).max() ?? 0.0)
                sections.append(activationSection)
            }
        }

        return ActivationVisualizationSnapshot(sections: sections, maxMagnitude: max(maxMagnitude, 0.000001))
    }

    func activeModel() -> ModelChoice {
        lock.lock()
        defer { lock.unlock() }
        return ModelChoice(rawValue: Int32(engine.activeModelKind())) ?? .mlp
    }

    private func buildInputSection(layer: Int, name: String) -> ActivationSectionSnapshot? {
        let channels = Int(engine.cnnInputChannels(CInt(layer)))
        let height = Int(engine.cnnInputHeight(CInt(layer)))
        let width = Int(engine.cnnInputWidth(CInt(layer)))

        guard channels > 0, height > 0, width > 0 else {
            return nil
        }

        var maps: [ActivationMapSnapshot] = []

        for channel in 0..<channels {
            var values: [Float] = []
            values.reserveCapacity(width * height)

            for y in 0..<height {
                for x in 0..<width {
                    values.append(Float(engine.cnnInputValue(CInt(layer), CInt(channel), CInt(y), CInt(x))))
                }
            }

            maps.append(ActivationMapSnapshot(
                id: "input-\(layer)-\(channel)",
                channel: channel,
                width: width,
                height: height,
                values: values
            ))
        }

        return ActivationSectionSnapshot(id: "input-\(layer)", name: name, width: width, height: height, maps: maps)
    }

    private func buildActivationSection(layer: Int, name: String) -> ActivationSectionSnapshot? {
        let channels = Int(engine.cnnActivationChannels(CInt(layer)))
        let height = Int(engine.cnnActivationHeight(CInt(layer)))
        let width = Int(engine.cnnActivationWidth(CInt(layer)))

        guard channels > 0, height > 0, width > 0 else {
            return nil
        }

        var maps: [ActivationMapSnapshot] = []

        for channel in 0..<channels {
            var values: [Float] = []
            values.reserveCapacity(width * height)

            for y in 0..<height {
                for x in 0..<width {
                    values.append(Float(engine.cnnActivationValue(CInt(layer), CInt(channel), CInt(y), CInt(x))))
                }
            }

            maps.append(ActivationMapSnapshot(
                id: "activation-\(layer)-\(channel)",
                channel: channel,
                width: width,
                height: height,
                values: values
            ))
        }

        return ActivationSectionSnapshot(id: "activation-\(layer)", name: name, width: width, height: height, maps: maps)
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
