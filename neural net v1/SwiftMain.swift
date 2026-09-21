//
//  SwiftMain.swift
//  neural net v1
//
//  Created by Oliver Homer on 31/08/2026.
//

import Foundation
import SwiftUI
import AppKit
import NeuralApp

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@MainActor final class KernelVisualizationStore: ObservableObject {
    @Published var snapshot: KernelVisualizationSnapshot?
}

@MainActor final class ActivationVisualizationStore: ObservableObject {
    @Published var snapshot: ActivationVisualizationSnapshot?
}

@main
struct SwiftMain: App  {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    @StateObject private var engineBox = EngineBox (engine: AppEngine())
    @StateObject private var kernelStore = KernelVisualizationStore()
    @StateObject private var activationStore = ActivationVisualizationStore()

    var body: some Scene {
        WindowGroup {
            HStack(spacing: 0) {
                SwiftUIView(engineBox: engineBox, kernelStore: kernelStore, activationStore: activationStore)
                    .frame(width: 320)

                Divider()

                DrawingPad(engineBox: engineBox, activationStore: activationStore)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }

        WindowGroup("CNN Kernels", id: "cnn-kernels") {
            KernelVisualizationWindow(snapshot: kernelStore.snapshot)
                .frame(minWidth: 720, minHeight: 520)
        }

        WindowGroup("CNN Activations", id: "cnn-activations") {
            ActivationVisualizationWindow(snapshot: activationStore.snapshot)
                .frame(minWidth: 820, minHeight: 620)
        }
    }
}

private struct KernelVisualizationWindow: View {
    let snapshot: KernelVisualizationSnapshot?

    var body: some View {
        Group {
            if let snapshot {
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        KernelLegend()

                        ForEach(snapshot.layers) { layer in
                            KernelLayerView(layer: layer, maxMagnitude: snapshot.maxMagnitude)
                        }
                    }
                    .padding(20)
                }
            } else {
                ContentUnavailableView("No CNN kernels yet", systemImage: "square.grid.3x3")
            }
        }
    }
}

private struct KernelLegend: View {
    var body: some View {
        HStack(spacing: 12) {
            Text("CNN Kernels")
                .font(.title2.weight(.semibold))

            Spacer()

            HStack(spacing: 6) {
                Rectangle()
                    .fill(Color.blue)
                    .frame(width: 14, height: 14)
                Text("Negative")
                Rectangle()
                    .fill(Color.red)
                    .frame(width: 14, height: 14)
                Text("Positive")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }
}

private struct KernelLayerView: View {
    let layer: KernelLayerSnapshot
    let maxMagnitude: Float

    private var columns: [GridItem] {
        [GridItem(.adaptive(minimum: 86), spacing: 12)]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(layer.name)
                    .font(.headline)
                Text("\(layer.outputChannels) output x \(layer.inputChannels) input, \(layer.kernelWidth)x\(layer.kernelHeight)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            LazyVGrid(columns: columns, alignment: .leading, spacing: 12) {
                ForEach(layer.kernels) { kernel in
                    KernelTileView(kernel: kernel, maxMagnitude: maxMagnitude)
                }
            }
        }
    }
}

private struct KernelTileView: View {
    let kernel: KernelSnapshot
    let maxMagnitude: Float

    private var columns: [GridItem] {
        Array(repeating: GridItem(.fixed(18), spacing: 2), count: kernel.width)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("O\(kernel.outputChannel) I\(kernel.inputChannel)")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)

            LazyVGrid(columns: columns, spacing: 2) {
                ForEach(kernel.values.indices, id: \.self) { index in
                    Rectangle()
                        .fill(color(for: kernel.values[index]))
                        .frame(width: 18, height: 18)
                }
            }
            .padding(4)
            .background(.quaternary)
            .clipShape(RoundedRectangle(cornerRadius: 4))
        }
    }

    private func color(for value: Float) -> Color {
        let strength = min(Double(abs(value) / maxMagnitude), 1.0)
        let opacity = 0.12 + strength * 0.88

        if value >= 0 {
            return Color.red.opacity(opacity)
        }

        return Color.blue.opacity(opacity)
    }
}

private struct ActivationVisualizationWindow: View {
    let snapshot: ActivationVisualizationSnapshot?

    var body: some View {
        Group {
            if let snapshot {
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        HStack(alignment: .firstTextBaseline) {
                            Text("CNN Activations")
                                .font(.title2.weight(.semibold))

                            Spacer()

                            Text("Brighter cells have larger activation")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        ForEach(snapshot.sections) { section in
                            ActivationSectionView(section: section)
                        }
                    }
                    .padding(20)
                }
            } else {
                ContentUnavailableView("No CNN activations yet", systemImage: "square.grid.3x3")
            }
        }
    }
}

private struct ActivationSectionView: View {
    let section: ActivationSectionSnapshot

    private var columns: [GridItem] {
        [GridItem(.adaptive(minimum: tileMinimumWidth), spacing: 14)]
    }

    private var tileMinimumWidth: CGFloat {
        section.width >= 28 ? 154 : 120
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(section.name)
                    .font(.headline)
                Text("\(section.maps.count) map\(section.maps.count == 1 ? "" : "s"), \(section.width)x\(section.height)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            LazyVGrid(columns: columns, alignment: .leading, spacing: 14) {
                ForEach(section.maps) { map in
                    ActivationMapTileView(map: map)
                }
            }
        }
    }
}

private struct ActivationMapTileView: View {
    let map: ActivationMapSnapshot

    private var cellSize: CGFloat {
        map.width >= 28 ? 5 : 8
    }

    private var localMaxMagnitude: Float {
        max(map.values.map(abs).max() ?? 0.0, 0.000001)
    }

    private var columns: [GridItem] {
        Array(repeating: GridItem(.fixed(cellSize), spacing: 1), count: map.width)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Channel \(map.channel)")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)

            LazyVGrid(columns: columns, spacing: 1) {
                ForEach(map.values.indices, id: \.self) { index in
                    Rectangle()
                        .fill(color(for: map.values[index]))
                        .frame(width: cellSize, height: cellSize)
                }
            }
            .padding(4)
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: 4))
        }
    }

    private func color(for value: Float) -> Color {
        let normalized = min(Double(abs(value) / localMaxMagnitude), 1.0)
        let boosted = pow(normalized, 0.55)
        return Color(hue: 0.34, saturation: 0.9, brightness: 0.08 + boosted * 0.92)
    }
}
