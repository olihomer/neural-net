import SwiftUI
import AppKit
import NeuralApp

@MainActor final class TrainingProgressModel: ObservableObject {
    @Published var epoch: Int32 = 0
    @Published var error: Double = 0.0
    @Published var runResult: Int32? = nil
}

// Keep a main-actor isolated weak global reference that the C/C++ callback can use to report progress.
@MainActor private weak var globalProgressModel: TrainingProgressModel?

public struct SwiftUIView: View {
    private enum FocusedField: Hashable {
        case hiddenLayerSize
        case epochs
        case trainingExamples
        case evaluationExamples
        case batchSize
        case learningRate
        case dropout
        case cnnConvChannels(Int)
        case cnnClassifierHiddenLayerSize
    }

    @StateObject private var progress = TrainingProgressModel()
    @ObservedObject private var engineBox: EngineBox
    @ObservedObject private var kernelStore: KernelVisualizationStore
    @ObservedObject private var activationStore: ActivationVisualizationStore
    @Environment(\.openWindow) private var openWindow
    @State private var settings = TrainingSettings()
    @State private var hiddenLayerSizeText = "128"
    @State private var epochsText = "50"
    @State private var trainingExamplesText = "1000"
    @State private var evaluationExamplesText = "100"
    @State private var batchSizeText = "50"
    @State private var learningRateText = "0.001"
    @State private var dropoutText = "0.1"
    @State private var cnnConvChannelTexts = ["8", "16", "32"]
    @State private var cnnClassifierHiddenLayerSizeText = "128"
    @State private var isTraining = false
    @State private var fileStatus = ""
    @FocusState private var focusedField: FocusedField?

    init(engineBox: EngineBox, kernelStore: KernelVisualizationStore, activationStore: ActivationVisualizationStore) {
        self.engineBox = engineBox
        self.kernelStore = kernelStore
        self.activationStore = activationStore
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Training Settings")
                .font(.headline)

            Picker("Model", selection: $settings.model) {
                ForEach(ModelChoice.allCases) { choice in
                    Text(choice.title).tag(choice)
                }
            }
            .pickerStyle(.segmented)

            integerField("Epochs", text: $epochsText, field: .epochs) {
                commitInteger($epochsText, to: $settings.epochs, range: 1...5000)
            }
            integerField("Training examples", text: $trainingExamplesText, field: .trainingExamples) {
                commitInteger($trainingExamplesText, to: $settings.trainingExamples, range: 1...60000)
            }
            integerField("Evaluation examples", text: $evaluationExamplesText, field: .evaluationExamples) {
                commitInteger($evaluationExamplesText, to: $settings.evaluationExamples, range: 1...10000)
            }
            integerField("Batch size", text: $batchSizeText, field: .batchSize) {
                commitInteger($batchSizeText, to: $settings.batchSize, range: 1...60000)
            }
            decimalField("Learning rate", text: $learningRateText, field: .learningRate) {
                commitDouble($learningRateText, to: $settings.learningRate, range: 0.0...2.0)
            }
            decimalField("Dropout", text: $dropoutText, field: .dropout) {
                commitDouble($dropoutText, to: $settings.dropout, range: 0.0...0.95)
            }

            if settings.model == .mlp {
                Text("MLP Shape")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                integerField("Hidden layer", text: $hiddenLayerSizeText, field: .hiddenLayerSize) {
                    commitInteger($hiddenLayerSizeText, to: $settings.hiddenLayerSize, range: 1...512)
                }

                Picker("Hidden activation", selection: $settings.hiddenActivation) {
                    ForEach(ActivationChoice.allCases) { choice in
                        Text(choice.title).tag(choice)
                    }
                }

                Picker("Output activation", selection: $settings.outputActivation) {
                    ForEach(ActivationChoice.allCases) { choice in
                        Text(choice.title).tag(choice)
                    }
                }
            }

            if settings.model == .cnn {
                Text("CNN Shape")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                ForEach(cnnConvChannelTexts.indices, id: \.self) { index in
                    integerField("Conv\(index + 1) channels", text: bindingForConvChannelText(at: index), field: .cnnConvChannels(index)) {
                        commitConvChannel(at: index)
                    }
                }

                HStack {
                    Button("Remove layer") {
                        removeConvLayer()
                    }
                    .disabled(cnnConvChannelTexts.count <= 1)

                    Button("Add layer") {
                        addConvLayer()
                    }
                }

                integerField("Classifier hidden", text: $cnnClassifierHiddenLayerSizeText, field: .cnnClassifierHiddenLayerSize) {
                    commitInteger($cnnClassifierHiddenLayerSizeText, to: $settings.cnnClassifierHiddenLayerSize, range: 1...512)
                }
            }

            Button(isTraining ? "Training..." : "Run Training") {
                commitNumericSettings()
                isTraining = true
                progress.runResult = nil
                fileStatus = ""
                globalProgressModel = progress
                let engineBox = engineBox
                let settings = settings

                // Run the heavy C++ work off the main actor so the UI can update.
                Task.detached {
                    let result = engineBox.runApp(settings: settings)
                    let kernelSnapshot = settings.model == .cnn ? engineBox.cnnKernelSnapshot() : nil
                    let activationSnapshot = settings.model == .cnn ? engineBox.cnnActivationSnapshot() : nil

                    await MainActor.run {
                        progress.runResult = result
                        isTraining = false

                        if let kernelSnapshot {
                            kernelStore.snapshot = kernelSnapshot
                        }

                        if let activationSnapshot {
                            activationStore.snapshot = activationSnapshot
                        }
                    }
                }
            }
            .disabled(isTraining)

            HStack {
                Button("Save") {
                    saveNetwork()
                }
                .disabled(isTraining)

                Button("Load") {
                    loadNetwork()
                }
                .disabled(isTraining)
            }

            HStack {
                Button("Show Kernels") {
                    openWindow(id: "cnn-kernels")
                }
                .disabled(kernelStore.snapshot == nil)

                Button("Show Activations") {
                    openWindow(id: "cnn-activations")
                }
                .disabled(activationStore.snapshot == nil)
            }

            if !fileStatus.isEmpty {
                Text(fileStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if isTraining {
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Training is running")
                }
            }

            Divider()

            Text("Result: \(progress.runResult.map(String.init) ?? "No result yet")")
            Text("Epoch: \(progress.epoch)")
            Text(String(format: "Error: %.6f", progress.error))

            Spacer()
        }
        .padding()
        .frame(minWidth: 280, alignment: .topLeading)
    }

    private func integerField(_ title: String, text: Binding<String>, field: FocusedField, onCommit: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
            TextField(title, text: text)
                .textFieldStyle(.roundedBorder)
                .focused($focusedField, equals: field)
                .onTapGesture {
                    focusedField = field
                }
                .onSubmit(onCommit)
        }
    }

    private func decimalField(_ title: String, text: Binding<String>, field: FocusedField, onCommit: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
            TextField(title, text: text)
                .textFieldStyle(.roundedBorder)
                .focused($focusedField, equals: field)
                .onTapGesture {
                    focusedField = field
                }
                .onSubmit(onCommit)
        }
    }

    private func commitNumericSettings() {
        commitInteger($hiddenLayerSizeText, to: $settings.hiddenLayerSize, range: 1...512)
        commitInteger($epochsText, to: $settings.epochs, range: 1...5000)
        commitInteger($trainingExamplesText, to: $settings.trainingExamples, range: 1...60000)
        commitInteger($evaluationExamplesText, to: $settings.evaluationExamples, range: 1...60000)
        commitInteger($batchSizeText, to: $settings.batchSize, range: 1...60000)
        commitDouble($learningRateText, to: $settings.learningRate, range: 0.0...2.0)
        commitDouble($dropoutText, to: $settings.dropout, range: 0.0...0.95)
        for index in cnnConvChannelTexts.indices {
            commitConvChannel(at: index)
        }
        commitInteger($cnnClassifierHiddenLayerSizeText, to: $settings.cnnClassifierHiddenLayerSize, range: 1...512)
    }

    private func bindingForConvChannelText(at index: Int) -> Binding<String> {
        Binding(
            get: {
                guard cnnConvChannelTexts.indices.contains(index) else {
                    return "1"
                }
                return cnnConvChannelTexts[index]
            },
            set: { newValue in
                guard cnnConvChannelTexts.indices.contains(index) else {
                    return
                }
                cnnConvChannelTexts[index] = newValue
            }
        )
    }

    private func commitConvChannel(at index: Int) {
        guard cnnConvChannelTexts.indices.contains(index) else {
            return
        }

        let text = bindingForConvChannelText(at: index)
        let value = Binding<Int>(
            get: {
                guard settings.cnnConvChannels.indices.contains(index) else {
                    return 1
                }
                return settings.cnnConvChannels[index]
            },
            set: { newValue in
                guard settings.cnnConvChannels.indices.contains(index) else {
                    return
                }
                settings.cnnConvChannels[index] = newValue
            }
        )

        commitInteger(text, to: value, range: 1...512)
    }

    private func addConvLayer() {
        cnnConvChannelTexts.append("32")
        settings.cnnConvChannels.append(32)
    }

    private func removeConvLayer() {
        guard cnnConvChannelTexts.count > 1 else {
            return
        }
        cnnConvChannelTexts.removeLast()
        settings.cnnConvChannels.removeLast()
    }

    private func saveNetwork() {
        let panel = NSSavePanel()
        panel.title = "Save Neural Network"
        panel.nameFieldStringValue = "network.nnet"
        panel.canCreateDirectories = true

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        fileStatus = engineBox.saveNetwork(to: url.path) ? "Saved network" : "Save failed"
    }

    private func loadNetwork() {
        let panel = NSOpenPanel()
        panel.title = "Load Neural Network"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        fileStatus = engineBox.loadNetwork(from: url.path) ? "Loaded network" : "Load failed"
    }

    private func commitInteger(_ text: Binding<String>, to value: Binding<Int>, range: ClosedRange<Int>) {
        guard let parsed = Int(text.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            text.wrappedValue = String(value.wrappedValue)
            return
        }

        let clamped = min(max(parsed, range.lowerBound), range.upperBound)
        value.wrappedValue = clamped
        text.wrappedValue = String(clamped)
    }

    private func commitDouble(_ text: Binding<String>, to value: Binding<Double>, range: ClosedRange<Double>) {
        guard let parsed = Double(text.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            text.wrappedValue = String(value.wrappedValue)
            return
        }

        let clamped = min(max(parsed, range.lowerBound), range.upperBound)
        value.wrappedValue = clamped
        text.wrappedValue = String(clamped)
    }
}

// MARK: - C/C++ progress callback bridge
@MainActor
private func applyProgressUpdate(epoch: Int32, error: Double) {
    globalProgressModel?.epoch = epoch
    globalProgressModel?.error = error
}

@Sendable
func progress_callback(newEpoch: Int32, newError: Double) {
    // Schedule the UI update onto the main actor without capturing main-actor state here
    Task { @MainActor in
        applyProgressUpdate(epoch: newEpoch, error: newError)
    }
}
