import SwiftUI
import NeuralApp

@MainActor final class TrainingProgressModel: ObservableObject {
    @Published var epoch: Int32 = 0
    @Published var error: Double = 0.0
    @Published var runResult: Int32? = nil
}

// Keep a main-actor isolated weak global reference that the C/C++ callback can use to report progress.
@MainActor private weak var globalProgressModel: TrainingProgressModel?

public struct SwiftUIView: View {
    private enum FocusedField {
        case hiddenLayerSize
        case epochs
        case trainingExamples
        case learningRate
    }

    @StateObject private var progress = TrainingProgressModel()
    @ObservedObject private var engineBox: EngineBox
    @State private var settings = TrainingSettings()
    @State private var hiddenLayerSizeText = "128"
    @State private var epochsText = "500"
    @State private var trainingExamplesText = "100"
    @State private var learningRateText = "0.5"
    @State private var isTraining = false
    @FocusState private var focusedField: FocusedField?

    init(engineBox: EngineBox) {
        self.engineBox = engineBox
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Training Settings")
                .font(.headline)

            integerField("Hidden layer", text: $hiddenLayerSizeText, field: .hiddenLayerSize) {
                commitInteger($hiddenLayerSizeText, to: $settings.hiddenLayerSize, range: 1...512)
            }
            integerField("Epochs", text: $epochsText, field: .epochs) {
                commitInteger($epochsText, to: $settings.epochs, range: 1...5000)
            }
            integerField("Training examples", text: $trainingExamplesText, field: .trainingExamples) {
                commitInteger($trainingExamplesText, to: $settings.trainingExamples, range: 1...60000)
            }
            decimalField("Learning rate", text: $learningRateText, field: .learningRate) {
                commitDouble($learningRateText, to: $settings.learningRate, range: 0.0...2.0)
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

            Button(isTraining ? "Training..." : "Run Training") {
                commitNumericSettings()
                isTraining = true
                progress.runResult = nil
                globalProgressModel = progress
                let engineBox = engineBox
                let settings = settings

                // Run the heavy C++ work off the main actor so the UI can update.
                Task.detached {
                    let result = engineBox.runApp(settings: settings)
                    await MainActor.run {
                        progress.runResult = result
                        isTraining = false
                    }
                }
            }
            .disabled(isTraining)

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
        commitDouble($learningRateText, to: $settings.learningRate, range: 0.0...2.0)
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
