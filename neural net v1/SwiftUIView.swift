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
    @StateObject private var progress = TrainingProgressModel()
    @ObservedObject private var engineBox: EngineBox

    init(engineBox: EngineBox) {
        self.engineBox = engineBox
    }

    public var body: some View {
        VStack(spacing: 12) {
            Button("Run C++ App") {
                // Register the model so the callback can update it
                globalProgressModel = progress
                let engineBox = engineBox

                // Run the heavy C++ work off the main actor so the UI can update.
                Task.detached {
                    let result = engineBox.runApp()
                    await MainActor.run {
                        progress.runResult = result
                    }
                }
            }
            Button("Test") {
                // Example manual update to verify UI binding
                progress.epoch += 1
                progress.error = Double.random(in: 0...1)
            }
            Text("Result: \(progress.runResult.map(String.init) ?? "No result yet")")
            Text("Epoch: \(progress.epoch)")
            Text(String(format: "Error: %.6f", progress.error))
        }
        .padding()
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
